import os
import pandas as pd
from pathlib import Path

def print_df_info(df):
    # 显示DataFrame的内容
    print("\n=== DataFrame内容 ===")
    print(df.head())  # 显示前5行
    print("\n=== DataFrame基本信息 ===")
    print(df.info())  # 显示数据类型和非空值统计
    print("\n=== DataFrame统计描述 ===")
    print(df.describe())  # 显示数值列的统计信息

parquet_dir_path = os.environ.get('PARQUET_DIR_PATH')
parquet_dir_path = Path(parquet_dir_path)

parquet_file_path = parquet_dir_path / 'train-00000-of-00082.parquet'
df = pd.read_parquet(parquet_file_path)
print_df_info(df)

# parquet_file_path = parquet_dir_path / 'train-00001-of-00082.parquet'
# df = pd.read_parquet(parquet_file_path)
# print_df_info(df)


# # 使用通配符匹配多个文件（适合连续编号的文件）
# files = sorted(parquet_dir_path.glob('train-0000*-of-00082.parquet'))
# # 读取并合并parquet文件
# df = pd.concat(
#     [pd.read_parquet(f) for f in files],
#     ignore_index=True
# )

# # 保存合并后的文件
# output_path = parquet_dir_path / "merged.parquet"
# df.to_parquet(output_path, index=False)
# print(f"\n=== 合并完成，保存至 {output_path} ===")
# print_df_info(df)