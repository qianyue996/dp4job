import os
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

load_dotenv()


class MCPClient:
    def __init__(self):
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        # 创建环境变量字典
        env = os.environ.copy()

        server_params = StdioServerParameters(
            command="python", args=[server_script_path], env=env  # 传递环境变量
        )

        self.stdio, self.write = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # 可用工具列表
        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]
        return available_tools

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
