'''
pyEchart生成图表分析
'''
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.charts import Graph
import webbrowser


def bar_temp():
    # V1 版本开始支持链式调用
    bar = (
        Bar()
            .add_xaxis(["衬衫", "毛衣", "领带", "裤子", "风衣", "高跟鞋", "袜子"])
            .add_yaxis("商家A", [114, 55, 27, 101, 125, 27, 105])
            .add_yaxis("商家B", [57, 134, 137, 129, 145, 60, 49])
            .set_global_opts(title_opts=opts.TitleOpts(title="某商场销售情况"))
    )
    bar.render()

def Graph_with_edge_options(nodes_data, links_data):
    Graph_tuopu = Graph()
    ## 配置大小
    Graph_tuopu.__init__(init_opts=opts.InitOpts(width="1100px", height="1500px"))
    Graph_tuopu.add(
        "",
        nodes_data,
        links_data,
        repulsion=4000,
        # edge_label=opts.LabelOpts(
        #     is_show=True, position="middle", formatter="{b} 的数据 {c}"
        # ),
        itemstyle_opts=opts.ItemStyleOpts(color="gray")
    )
    Graph_tuopu.set_global_opts(
        title_opts=opts.TitleOpts(title="topology_edges_node.json")
    )
    Graph_tuopu.render("graph_with_edge_options.html")
    webbrowser.open("../../Game/suning2020/graph_with_edge_options.html")
    print("OK")


if __name__ == '__main__':
    nodes_data = [
        opts.GraphNode(name="结点1", symbol_size=20,category=1,
                       label_opts=opts.LabelOpts(color="red", background_color="yellow"),
                       ),
        opts.GraphNode(name="结点2", symbol_size=20),
        opts.GraphNode(name="结点3", symbol_size=20),
        opts.GraphNode(name="结点4", symbol_size=20),
        opts.GraphNode(name="结点5", symbol_size=20),
        opts.GraphNode(name="结点6", symbol_size=20),
    ]
    links_data = [
        opts.GraphLink(source="结点1", target="结点2", value=2,
                       linestyle_opts=opts.LineStyleOpts(color="red")
                       ),
        opts.GraphLink(source="结点2", target="结点3", value=3),
        opts.GraphLink(source="结点3", target="结点4", value=4),
        opts.GraphLink(source="结点4", target="结点5", value=5),
        opts.GraphLink(source="结点5", target="结点6", value=6),

    ]
    Graph_with_edge_options(nodes_data, links_data)

"""
 上述情况下，我们可以自定义画布的大小，并且生成对象，现在我们需要
    1,  我们需要将json的数据导入并且展示成拓扑图 
    2,  json数据格式中，我们随机标记root节点
    3,  再拓扑图中
"""
