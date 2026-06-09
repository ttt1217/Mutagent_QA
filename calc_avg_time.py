import json

def calc_average_elapsed_time(json_path="summary.json"):
    """计算 summary.json 中问答的平均耗时"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    results = data.get("results", [])
    if not results:
        print(" 未找到 results 字段或为空")
        return

    total_time = 0.0
    count = 0

    for item in results:
        elapsed = item.get("elapsed_time")
        if isinstance(elapsed, (int, float)):
            total_time += elapsed
            count += 1
        else:
            print(f"跳过无效耗时记录: {item.get('question', '未知问题')}")

    if count == 0:
        print("⚠没有有效的耗时数据可计算")
        return

    avg_time = total_time / count
    print(f"共统计 {count} 条问答")
    print(f"平均耗时: {avg_time:.2f} 秒")

    return avg_time


if __name__ == "__main__":
    calc_average_elapsed_time("summary.json")
