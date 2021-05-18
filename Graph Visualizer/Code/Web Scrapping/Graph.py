import requests
from bs4 import BeautifulSoup
import re

graph_res = requests.get('https://www.geeksforgeeks.org/graph-data-structure-and-algorithms/')
#print(graph_res.status_code)
src = graph_res.content
soup = BeautifulSoup(src, 'lxml')

graph_info = []
for p_tag in soup.find_all('p',text = re.compile('([A Graph])+')):
    graph_info.append(p_tag.getText())

graph_img = soup.find_all('img', alt="")
print(graph_img)
definition = graph_info[0:2]
print(definition)