def bfs(graph,s_node):
    que = [s_node]
    arr = []
    while que:
        nod = que.pop(0)
        if nod not in arr:
            arr.append(nod)
            print(f"{nod}")
            for i in graph[nod]:
                if i not in arr:
                    que.append(i)
graph = {"A":["A","D","B","E"],
         "B":["A","C","F"],
         "C":["B","E","D"],
         "D":["A","E","C"],
         "E":["A","C"],
         "F":["B"],
         }

start = "D"
d = bfs(graph,start)
# print("BFS is:",d)
                             
        