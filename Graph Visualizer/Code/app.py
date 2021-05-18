import io
import base64
from flask import Flask,render_template,request,url_for,redirect,Response,session,make_response
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue
from random import randint, uniform
import sys
import datetime
import networkx as nx
import matplotlib as mpl
from matplotlib import animation,rc

rc('animation', html='html5')
mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\monis\\Downloads\\ffmpeg\\ffmpeg\\bin\\ffmpeg.exe'

app = Flask(__name__)
app.permanent_session_lifetime = datetime.timedelta(days=365)
app.secret_key = '12345'
graph = None
pos = None
fig, ax = plt.subplots(figsize=(6, 4))
all_edges = None
spt_edges = None
name=None
weight=0
dijkstra_map = {}

inf = sys.maxsize


def bellmanFord(G, source, pos):
    global weight
    weight=0
    V = len(G.nodes())
    dist = []
    parent = [None] * V  # parent[i] will hold the node from which i is reached to, in the shortest path from source

    for i in range(V):
        dist.append(inf)

    parent[source] = -1;  # source is itself the root, and hence has no parent
    dist[source] = 0;
    for i in range(V - 1):
        for u, v, d in G.edges(data=True):  # Relaxation
            if dist[u] + d['weight'] < dist[v]:  # Relaxation Equation
                dist[v] = d['weight'] + dist[u]
                parent[v] = u

    # marking the shortest path from source to each of the vertex with red, using parent[]
    for X in range(V):
        if parent[X] != -1:  # ignore the parent of root node
            if (parent[X], X) in G.edges():
                print(spt_edges)
                spt_edges.add(tuple([parent[X], X, G[parent[X]][X]['weight']]))
                #yield spt_edges
                weight+=G[parent[X]][X]['weight']
    final = set()
    print(dist,"distance")
    for e in spt_edges:
        final.add(e)
        print(final, "Final!!")
        yield final


def random_node(NUM_NODES):
    return randint(1, NUM_NODES)

def prims(NUM_NODES):
    pqueue = PriorityQueue()
    global graph,weight
    print(list(graph.nodes))
    print(list(graph.edges))
    edges_in_mst = set()
    nodes_on_mst = set()

    start_node = random_node(NUM_NODES)
    for neighbor in graph.neighbors(start_node):
        edge_data = graph.get_edge_data(start_node, neighbor)
        edge_weight = edge_data["weight"]
        pqueue.put((edge_weight, (start_node, neighbor)))

    # Loop until all nodes are in the MST
    while len(nodes_on_mst) < NUM_NODES:
        _, edge = pqueue.get(pqueue)
        print(_, edge)

        if edge[0] not in nodes_on_mst:
            new_node = edge[0]
        elif edge[1] not in nodes_on_mst:
            new_node = edge[1]
        else:
            # skip if nodes already exist
            continue

        # Every time a new node is added to the priority queue, add
        # all edges that it sits on to the priority queue.
        for neighbor in graph.neighbors(new_node):
            edge_data = graph.get_edge_data(new_node, neighbor)
            edge_weight = edge_data["weight"]
            pqueue.put((edge_weight, (new_node, neighbor)))

        # Add this edge to the MST.
        edges_in_mst.add(tuple(sorted(edge)))
        nodes_on_mst.add(new_node)

        # Yield edges in the MST to plot.
        yield edges_in_mst
    weight = 0
    for e in edges_in_mst:
        x = list(e)
        edge_data = graph.get_edge_data(x[0],x[1])
        edge_weight = edge_data["weight"]
        weight+=edge_weight

def define_graph(graph, NUM_NODES,message):
    edges = message.split(',')
    for e in edges:
        n1,n2,ew = e.split(':')
        graph.add_edge(int(n1),int(n2),weight=int(ew))

def update(mst_edges):
    #print("Updated",mst_edges)
    global pos,fig,ax,graph,all_edges
    ax.clear()
    labels = nx.get_edge_attributes(graph,'weight')
    nodes ={}
    for node in graph.nodes():
        nodes[node] = node
    nx.draw_networkx_nodes(graph, pos, node_size=300, ax=ax, node_color="blue", linewidths=5)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    nx.draw_networkx_labels(graph, pos, nodes, font_color='white')
    nx.draw_networkx_edges(
        graph, pos, edgelist=all_edges-mst_edges, alpha=0.25,
        edge_color='blue', width=1, ax=ax
    )
    nx.draw_networkx_edges(
        graph, pos, edgelist=mst_edges, alpha=1.0,
        edge_color='red', width=1, ax=ax
    )
def minDistance(dist, sptSet, V):
    global dijkstra_map
    min = sys.maxsize  # assigning largest numeric value to min
    for v in range(V):
        if sptSet[v] == False and dist[v] <= min:
            min = dist[v]
            min_index = v
    dijkstra_map[min_index] = min
    return min_index


# function that performs dijsktras algorithm
def dijsktras(G, source, pos):
    global weight
    weight=0
    V = len(G.nodes())
    dist = []
    parent = [None] * V  # parent[i] will hold the node from which i is reached to, in the shortest path from source
    sptSet = []  # sptSet[i] will hold true if vertex i is included in shortest path tree

    for i in range(V):
        dist.append(sys.maxsize)
        sptSet.append(False)
    dist[source] = 0
    parent[source] = -1  # source is itself the root, and hence has no parent
    for count in range(V):
        u = minDistance(dist, sptSet, V)  # pick the minimum distance vertex from the set of vertices
        sptSet[u] = True
        # update the vertices adjacent to the picked vertex
        for v in range(V):
            if (u, v) in G.edges():
                if sptSet[v] == False and dist[u] != sys.maxsize and dist[u] + G[u][v]['weight'] < dist[v]:
                    dist[v] = dist[u] + G[u][v]['weight']
                    parent[v] = u
                    #print(u,v,"SRC and DEST")
    # marking the shortest path from source to each of the vertex with red, using parent[]
    for X in range(V):
        if parent[X] != -1:  # ignore the parent of root node
            if (parent[X], X) in G.edges():
                #print(spt_edges)
                spt_edges.add(tuple([parent[X],X,G[parent[X]][X]['weight']]))
                weight+=G[parent[X]][X]['weight']
                #nx.draw_networkx_edges(G, pos, edgelist=[(parent[X], X)], width=2.5, alpha=1, edge_color='red')
                #yield spt_edges
    final = set()

    for e in spt_edges:
        final.add(e)
        #print(final, "Final!!")
        yield final

def do_nothing():
    # FuncAnimation requires an initialization function. We don't
    # do any initialization, so we provide a no-op function.
    pass

@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method =="POST":
        meth=request.form["method"]
        if meth=='Prims':
            return redirect(url_for('Prims'))
        if meth=='BellManFord':
            return redirect(url_for('BellManFord'))
        if meth=='Dijkstra':
            return redirect(url_for('Dijkstra'))
    else:
        return render_template('index.html')

@app.route('/Prims',methods=['GET','POST'])
def Prims():
    global pos,fig,ax,graph,all_edges,name
    if request.method =="POST":
        message=request.form["message"]
        print(message)
        NUM_NODES = int(request.form["nodes"])
        graph = nx.Graph()
        define_graph(graph,NUM_NODES, message)
        #print(list(graph.nodes))
        pos = nx.spring_layout(graph)

        all_edges = set(
            tuple(sorted((n1, n2))) for n1, n2 in graph.edges()
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ani = animation.FuncAnimation(
                fig,
                update,
                init_func=do_nothing,
                frames=prims(NUM_NODES),
                interval=600,
                repeat=False
            )
        FFwriter = animation.FFMpegWriter(fps=1)
        ani.save('static/animation.mp4', writer=FFwriter)
        name="Prim's"
        res = make_response(render_template('plot.html',name="Prim's", weight=weight))
        res.set_cookie('NUM_NODES', str(NUM_NODES), domain='127.0.0.1')
        res.set_cookie('message', str(message), domain='127.0.0.1')
        return res
        #return redirect(url_for('plot'))
        #return render_template('plot.html',name="Prim's")
    else:
        message=request.cookies.get('message')
        NUM_NODES=request.cookies.get('NUM_NODES')
        
        url4 = 'https://www.programiz.com/dsa/prim-algorithm'
        response4=requests.get(url4) 
        soup4=BeautifulSoup(response4.content,'html.parser') 
        result1 = soup4.select('#node-1056 > div > p:nth-child(5)')
        result1[0].text
        result2 = soup4.select('#node-1056 > div > p:nth-child(6)')
        result2[0].text
        result3 = soup4.select('#node-1056 > div > p:nth-child(7)')
        result3[0].text
        result4 = result1[0].text+result2[0].text+result3[0].text
        points=[]
        parent2 = soup4.select('#node-1056 > div > ol')
        for li in parent2[0].findAll('li'):
            points.append(li.text)

        return render_template('Prims.html',val=[message,NUM_NODES],result4=str(result4), points=points)

@app.route('/BellManFord',methods=['GET','POST'])
def BellManFord():
    global pos,fig,ax,graph,all_edges,spt_edges,name
    if request.method =="POST":
        message=request.form["message"]
        print(message)
        NUM_NODES = int(request.form["nodes"])
        graph = nx.DiGraph()

        define_graph(graph, NUM_NODES, message)
        pos = nx.spring_layout(graph)
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = nx.get_edge_attributes(graph, 'weight')
        all_edges = set(
            tuple((n1, n2)) for n1, n2 in graph.edges()
        )
        spt_edges= set()
        source = int(request.form["source"])
        ani = animation.FuncAnimation(
            fig,
            update,
            init_func=do_nothing,
            frames=bellmanFord(graph, source, pos),
            interval=1000,
            repeat=False
        )
        FFwriter = animation.FFMpegWriter(fps=1)
        ani.save('static/animation.mp4', writer=FFwriter)
        print("Video Created!")
        name='BellmanFord'
        return redirect(url_for('plot'))
        #return render_template('plot.html', name="BellmanFord")
    else:
        url="https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm"
        r = requests.get(url)
        soup = BeautifulSoup(r.content) 
        table = soup.find('pre') 
        print(table)

        url3 = 'https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm'
        response3=requests.get(url3) 
        soup3=BeautifulSoup(response3.content,'html.parser') 
        all_results = soup3.select('#mw-content-text > div.mw-parser-output > p:nth-child(30)')
        block1 = all_results[0].text.replace('\n', ' ')

        parent = soup3.select('#mw-content-text > div.mw-parser-output > ol')
        lis = []
        for li in parent[0].findAll('li'):
            #print(li.text)
            lis.append(li.text)

        return render_template('BellManFord.html',table=str(table), block1=str(block1), lis=lis)

@app.route('/Dijkstra',methods=['GET','POST'])
def Dijkstra():
    global pos,fig,ax,graph,all_edges,spt_edges,name,dijkstra_map
    if request.method =="POST":
        message=request.form["message"]
        nodes=request.form["nodes"]
        source=request.form["source"]
        session["Dijkstra"]={"message":message,"nodes":nodes,"source":source}
        # if "save" in request.form: 
        #     #print("Inside session",session["Dijkstra"])
        #     pass
        # else:
        #     print("Not an yes")
        #if save=="yes":

        #if(message.save)
        print(message)
        NUM_NODES = int(nodes)
        graph = nx.Graph()

        define_graph(graph, NUM_NODES,message)
        pos = nx.spring_layout(graph)
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = nx.get_edge_attributes(graph, 'weight')
        all_edges = set(
            tuple((n1, n2)) for n1, n2 in graph.edges()
        )
        spt_edges = set()
        source = int(source)
        ani = animation.FuncAnimation(
            fig,
            update,
            init_func=do_nothing,
            frames=dijsktras(graph,source,pos),
            interval=1000,
            repeat=False
        )
        FFwriter = animation.FFMpegWriter(fps=1)
        ani.save('static/animation.mp4', writer=FFwriter)
        print("Video Created!")
        print(dijkstra_map,"distance")
        dijkstra_map = {} #Clear the dictionary
        name="Dijkstra's"
        return redirect(url_for('plot'))
    else:
        val=""
        if "Dijkstra" in session:
            message=session["Dijkstra"]["message"]
            nodes=session["Dijkstra"]["nodes"]
            source=session["Dijkstra"]["source"]
            val=[message,nodes,source]
            print("9y3i7rgfd",nodes,source)
        return render_template('dijkstra.html',val=val)

@app.route('/plot',methods=['GET','POST'])
def plot():
    if(name=="Prim's"):
        return render_template('plot.html', name=name, weight=weight)
    else:
        return render_template('plot.html', name=name)

if __name__=="__main__":
    app.run(debug=True)
