import openai
from .base_vlm import BaseVLM, GenerateResponseResult
import os
import base64
import json
import io
from PIL import Image

class VLM(BaseVLM):
    """
    A VLM implementation for local vLLM servers via OpenAI-compatible Chat Completions API.
    """

    def __init__(self, config: dict, cache: bool = True):
        super().__init__(config=config, cache=cache)
        self.config = config

        base_url = os.environ.get("VLLM_BASE_URL", config.get("base_url", "http://localhost:8000/v1"))
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def load_video(self, video_path: str) -> list:
        np_frames = super().load_video(video_path)

        base64_frames = []
        for frame in np_frames:
            pil_img = Image.fromarray(frame)
            buff = io.BytesIO()
            pil_img.save(buff, format="PNG")
            base64_frame = base64.b64encode(buff.getvalue()).decode('utf-8')
            base64_frames.append(base64_frame)
        return base64_frames

    def load_image(self, image_path: str) -> str:
        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')
        return base64_image

    def create_messages(self, inputs, sys=None):
        messages = []
        if sys is not None:
            messages.append({"role": "system", "content": sys})

        user_content = []
        for part in inputs:
            if isinstance(part, str) and part.endswith(".mp4"):
                frames = self.load_video(part)
                for frame in frames:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{frame}"}
                    })
            elif isinstance(part, str) and part.endswith(".png"):
                image = self.load_image(part)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"}
                })
            elif isinstance(part, str) and (part.endswith(".jpg") or part.endswith(".jpeg")):
                image = self.load_image(part)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"}
                })
            else:
                user_content.append({
                    "type": "text",
                    "text": str(part)
                })

        messages.append({"role": "user", "content": user_content})
        return messages

    def generate_response(self, messages: str, output_path=None, return_token_counts=False) -> GenerateResponseResult:
        reasoning_text = ""
        if self.cache and output_path is not None and os.path.exists(output_path):
            with open(output_path, "r") as f:
                response = f.read()
            full_response = None
        else:
            full_response = self._chat_create(messages)
            msg = full_response.choices[0].message
            raw_content = msg.content or ""

            # Reasoning extraction:
            # - If a vLLM reasoning-parser is active it fills `reasoning_content`.
            # - Otherwise (no parser) dedicated thinking models emit
            #   "<reasoning>...</think><answer>" where the opening <think> is
            #   pre-filled in the prompt, so we split on </think> ourselves.
            # The parser's field is called `reasoning_content` on some vLLM
            # builds and `reasoning` on others - accept either.
            reasoning_text = getattr(msg, "reasoning_content", "") or getattr(msg, "reasoning", "") or ""
            if "</think>" in raw_content:
                think_part, _, after = raw_content.partition("</think>")
                if not reasoning_text:
                    reasoning_text = think_part.replace("<think>", "").strip()
                raw_content = after

            print(f"[DEBUG] reasoning chars: {len(reasoning_text)}", flush=True)
            print(f"[DEBUG] reasoning first 300: {reasoning_text[:300]!r}", flush=True)
            print(f"[DEBUG] content first 500 chars: {raw_content[:500]!r}", flush=True)
            response = raw_content.strip()

            # Strip ```json wrapper
            if response.startswith("```json"):
                response = response.split("\n")[1:-1]
                response = "\n".join(response)

        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            pass

        if output_path is not None:
            if output_path.endswith(".json"):
                with open(output_path, "w") as f:
                    json.dump(response, f, indent=4)
            else:
                with open(output_path, "w") as f:
                    if isinstance(response, dict):
                        json.dump(response, f, indent=4)
                    else:
                        f.write(str(response))

        response_object = GenerateResponseResult(
            response_json=response,
            response_text=response if isinstance(response, str) else json.dumps(response),
            thinking_text=reasoning_text
        )

        if return_token_counts:
            response_object.token_counts = (
                self.extract_token_counts(full_response) if full_response is not None
                else {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            )
        return response_object

    def generate_with_tools(self, messages, tools):
        """OpenAI-format tool-calling for the MCP loop. Returns the full
        chat completion response.
        """
        return self._chat_create(messages, tools=tools)
        


    def _chat_create(self, messages, tools=None):
        kwargs = {
            "model": self.config["model"],
            "messages": messages,
            "max_tokens": self.config.get("max_tokens", 20000),
            "temperature": self.config.get("temperature", 0.6),
            "top_p": self.config.get("top_p", 1.0),
        }

        reasoning_effort = self.config.get("reasoning_effort")
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        chat_template_kwargs = self.config.get("chat_template_kwargs")
        if chat_template_kwargs:
            kwargs["extra_body"] = {
                "chat_template_kwargs": chat_template_kwargs,
            }

        if tools is not None:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)
