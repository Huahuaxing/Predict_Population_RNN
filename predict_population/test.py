# 爬取网站的数据，并以excel表格（年份，总人口）的形式保存到当前项目目录

import requests

def spider_population():

    url = 'https://data.stats.gov.cn/easyquery.htm?m=QueryData&dbcode=hgnd&rowcode=zb&colcode=sj&wds=%5B%5D&dfwds=%5B%7B%22wdcode%22%3A%22sj%22%2C%22valuecode%22%3A%22last50%22%7D%5D&k1=1718356740871'

    population_dict = {}

    response1 = requests.get(url)

    population_dict = get_population_info(population_dict, response1.json())


    # save_excel(population_dict)

def get_population_info(population_dict, json_obj):
    """"datanodes":[{"code":"zb.A030101_sj.2023",
                     "data":{"data":140967.0,"dotcount":0,"hasdata":true,"strdata":"140967"},
                     "wds":[{"valuecode":"A030101","wdcode":"zb"},{"valuecode":"2023","wdcode":"sj"}]},"""
    datanodes = json_obj['returndata']['datanodes']

    for node in datanodes:
        year = node['code'][-4:]
        data = node['data']['data']
        if year in population_dict.keys():
            break       # 避免重复添加其他内容，我们只需要总人口就够了
        else:
            # population_dict[year] = [int(year), data]       # {'year':year, data, }
            print(year)
    return population_dict



# def save_excel(population_dict):


def main():
    spider_population()


if __name__ == '__main__':
    main()

