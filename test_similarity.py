#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.similarity_calculation import TableSimilarityCalculator

def test_similarity_calculation():
    """测试相似度计算功能"""
    
    print("开始测试相似度计算功能...")
    
    try:
        # 初始化相似度计算器
        calculator = TableSimilarityCalculator()
        
        # 测试用的新表格描述
        test_description = "党员信息表 包含表头：姓名,性别,民族,学历,出生日期,身份证号,年龄,入党时间,转正时间,联系电话,党龄"
        
        print(f"\n测试表格描述: {test_description}")
        
        # 查找最佳匹配
        results = calculator.get_best_matches(test_description, top_n=3)
        
        if results['success']:
            print("\n=== 相似度分析结果 ===")
            print(results['formatted_output'])
            
            # 显示匹配结果的详细信息
            print("\n前3个最相似的表格：")
            for i, match in enumerate(results['matches'], 1):
                print(f"{i}. {match['table_name']} - {match['similarity_formatted']} 相似度")
                
        else:
            print(f"查找相似表格失败: {results['error']}")
            
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_similarity_calculation()