import requests
import pandas as pd

# 人口数excel文件的保存路径
POPULATION_EXCEL_PATH = 'population_data.xlsx'


def spider_population():
    """
    数据字符串参数：    zb：指标   sj：时间   A0301：总人口   LAST50：近七十年
    爬取近七十年人口数据
    """
    # 总人口
    dfwds1 = '[{"wdcode": "sj", "valuecode": "LAST50"}, {"wdcode": "zb", "valuecode": "A0301"}]'  # json数组，包含两个字典
    url = 'http://data.stats.gov.cn/easyquery.htm?m=QueryData&dbcode=hgnd&rowcode=sj&colcode=zb&wds=[]&dfwds={}'

    population_dict = {

    }

    response1 = requests.get(url.format(dfwds1))  # url.format(dfwds) 会将 dfwds 的值插入到 url 的占位符 {} 中，形成完整的请求 URL。

    population_dict = get_population_info(population_dict, response1.json())

    save_excel(population_dict)


def get_population_info(population_dict, json_obj):

    """
    datanodes = {"code":"zb.A030101_sj.2023",
                 "data":{"data":140967.0,"dotcount":0,"hasdata":true,"strdata":"140967"},
                 "wds":[{"valuecode":"A030101","wdcode":"zb"},{"valuecode":"2023","wdcode":"sj"}]}
     """
    datanodes = json_obj['returndata']['datanodes']

    for node in datanodes:
        # 获取年份，[-4:]表示从倒数第四个字符开始取到末尾，即年份四位
        year = node['code'][-4:]
        # 数据数值
        data = node['data']['data']
        if year in population_dict.keys():
            # population_dict[year].append(data)
            break
        else:
            population_dict[year] = [int(year), data]
        # print(population_dict)

    return population_dict


def save_excel(population_dict):
    df = pd.DataFrame(population_dict).T[::-1]
    # df.columns = ['年份', '年末总人口(万人)', '男性人口(万人)', '女性人口(万人)', '城镇人口(万人)', '乡村人口(万人)']
    df.columns = ['年份', '总人口）']
    # print(df)
    writer = pd.ExcelWriter(POPULATION_EXCEL_PATH)
    df.to_excel(excel_writer=writer, index=False, sheet_name='中国近十年人口数据')
    writer.close()


if __name__ == '__main__':
    spider_population()
