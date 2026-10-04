import json

from .base_vlm import AssistantTurn, ToolCall
from .openai import VLM as OpenAIVLM


class VLM(OpenAIVLM):
    """
    OpenAI reasoning models (GPT-5.x) driving the MCP loop through the Responses API.

    Chat Completions would be the shorter path, but it drops the reasoning traces
    between turns - over 40 MCP iterations the model would restart its own train of
    thought every turn. The Responses API keeps them: `store=False` plus
    `include=["reasoning.encrypted_content"]` returns the reasoning as encrypted
    items that we hand straight back in the next request, without server-side state.

    Message building (create_messages, load_image, load_video) and the plain
    generate_response path are inherited unchanged from the Chat/Responses VLM.
    """

    def build_tool_specs(self, mcp_tools):
        """Responses tools are flat - name and parameters sit at the top level
        instead of inside a nested "function" object.

        strict=False because the MCP schemas are not strict-mode compatible: they
        have optional parameters and no "additionalProperties": false.
        """
        specs = [
            {"type": "function",
             "name": t.name,
             "description": t.description,
             "parameters": t.inputSchema,
             "strict": False}
            for t in mcp_tools
        ]

        finish = self.FINISH_TOOL["function"]        # chat-shaped constant in base_vlm
        specs.append({"type": "function",
                      "name": finish["name"],
                      "description": finish["description"],
                      "parameters": finish["parameters"],
                      "strict": False})
        return specs

    def request_tool_turn(self, messages, tool_specs):
        create_args = {
            "model": self.config["model"],
            "input": messages,
            "tools": tool_specs,
            "tool_choice": "auto",
            "store": False,
            "max_output_tokens": self.config.get("max_tokens", 20000),
        }

        # temperature/top_p are rejected by reasoning models - effort is the only knob
        if "reasoning_level" in self.config:
            create_args["reasoning"] = {"effort": self.config["reasoning_level"],
                                        "summary": "auto"}
            create_args["include"] = ["reasoning.encrypted_content"]

        response = self.client.responses.create(**create_args)

        # call_id (not id) is what links a call to its function_call_output
        tool_calls = [
            ToolCall(id=item.call_id,
                     name=item.name,
                     args=json.loads(item.arguments or "{}"))
            for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]

        # the whole output list goes back into the conversation: the reasoning items
        # carry the encrypted traces the next turn needs, so they must not be filtered
        return AssistantTurn(
            messages=list(response.output),
            tool_calls=tool_calls,
            usage=self.extract_token_counts(response),
        )

    def create_tool_result_message(self, tool_call_id, content):
        """Responses has no "tool" role; a result is an item referencing the call."""
        return {"type": "function_call_output",
                "call_id": tool_call_id,
                "output": content}

    def create_image_user_message(self, images):
        """Same as the chat version but in Responses shape: input_image with the
        data URI as a plain string instead of a nested image_url object.
        """
        content = [{"type": "input_image", "image_url": f"data:{mime};base64,{data}"}
                   for mime, data in images]
        return {"role": "user", "content": content}

    def extract_token_counts(self, response):
        """Token usage for one Responses call. cached_input_tokens is reported so a
        pilot run shows how much of the resent prefix actually hits the cache - the
        dominant cost factor of the MCP loop.
        """
        usage = getattr(response, "usage", None)
        if not usage:
            return {}

        output_details = getattr(usage, "output_tokens_details", None)
        input_details = getattr(usage, "input_tokens_details", None)

        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "thinking_tokens": getattr(output_details, "reasoning_tokens", 0) or 0,
            "cached_input_tokens": getattr(input_details, "cached_tokens", 0) or 0,
        }
