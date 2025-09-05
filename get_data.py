# 爬取网站数据并以excel形式（年份，总人口）保存到当前目录中

import requests
import pandas as pd

def spider_data():

    dfwds1 = '[{"wdcode":"sj","valuecode":"last50"}, {"wdcode":"zb", "valuecode": "A0301"}]'
    url = 'https://data.stats.gov.cn/easyquery.htm?m=QueryData&dbcode=hgnd&rowcode=zb&colcode=sj&wds=[]&dfwds={}'

    response = requests.get(url.format(dfwds1))
    # print(response.json())

    population_dict = {}                                            # {'2023' : [2023, 11111]}

    population_dict = get_data_info(population_dict, response.json())
    # print(population_dict)

    save_excel(population_dict)


def get_data_info(population_dict, json_obj):
    """
    "returncode":200,
    "returndata":{"datanodes":[{"code":"zb.A030101_sj.2023",
                                "data":{"data":140967.0,"dotcount":0,"hasdata":true,"strdata":"140967"},
                                "wds":[{"valuecode":"A030101","wdcode":"zb"},{"valuecode":"2023","wdcode":"sj"}]},"""
    # 提取出来datanodes部分，datanodes是个字典，值是一个数组，数组里面是一个个的字典每个字典又有三个键值（code,data,wds）
    datanodes = json_obj['returndata']['datanodes']         # 定义了一个datanodes数组
    
    # 循环提取datanodes的数据到population_dict里
    for node in datanodes:

        year = node['code'][-4:]
        data = node['data']['data']

        if year in population_dict.keys():
            break                                           # 只输入人口数据所以只按年份循环一轮，第二轮终止循环
        else:
            population_dict[year] = [int(year), data]

    return population_dict


def save_excel(population_dict):
    # 首先使用pandas把字典转换成DataFrame形式
    df = pd.DataFrame(population_dict).T[::-1]
    df.columns = ['年份', '总人口']                            # 要把横着的表转换为竖着的表

    # 保存为excel形式
    writer = pd.ExcelWriter('population.xlsx')
    df.to_excel(excel_writer=writer, index=False, sheet_name='中国近五十年人口数据')
    writer.close()




def main():
    spider_data()


if __name__ == '__main__':
    # 爬取成功！
    main()