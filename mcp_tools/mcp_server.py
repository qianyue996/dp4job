import datetime

from mcp.server.fastmcp import FastMCP
from rag_client import RAG_SEARCH

mcp = FastMCP("Agent Tools")
local_rag_search = RAG_SEARCH()

@mcp.tool()
async def rag_search() -> str:
    """
    使用检索增强生成搜索就业岗位信息，这里能获得岗位信息。

    Returns:
        包含岗位信息的字符串
    """

    return local_rag_search


@mcp.tool()
async def get_current_time():
    """
    获取当前的日期和时间，并将其分解为年、月、日、时、分、秒。

    返回:
        dict: 一个字典，包含以下键值对:
              'year' (int): 当前年份 (例如: 2023)
              'month' (int): 当前月份 (1-12)
              'day' (int): 当前日期 (1-31)
              'hour' (int): 当前小时 (0-23)
              'minute' (int): 当前分钟 (0-59)
              'second' (int): 当前秒数 (0-59)
    """
    now = datetime.datetime.now()  # 获取当前日期和时间对象

    datetime_components = {
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "second": now.second,
    }
    return datetime_components


# 示例用法
if __name__ == "__main__":
    mcp.run(transport="stdio")
