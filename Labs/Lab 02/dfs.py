def dfs(graph,s_node,arr=None):
    if arr is None:
        arr = []
    arr.append(s_node)
    print(s_node,end=' ')
    for i in graph[s_node]:
        if i not in arr:
            dfs(graph,i,arr)
    return arr            
graph = {"A":["A","D","B","E"],
         "B":["A","C","F"],
         "C":["B","E","D"],
         "D":["A","E","C"],
         "E":["A","C"],
         "F":["B"],
         }

start = "F"
d = dfs(graph,start)