class CityData:
    def __init__(self, name, outconcount, outCons):
        self.name = name
        self.outconcount = outconcount
        self.outCons = outCons
        self.seen = False
        self.predecessor = -1

def directed(t_city, pa, pair, city_names):
    city_to_index = {name: i for i, name in enumerate(city_names)}
    arr = [[] for _ in range(t_city)]

    for city1, city2 in pair:
        arr[city_to_index[city1]].append(city_to_index[city2])
    return arr

# Main function
if __name__ == "__main__":
    t_city = 8
    pa = 10
    city_names = ["Lahore", "Jazira", "Kasur", "Sahiwal", "Okara", "Bakhar", "Khosab", "Jhang"]
    pair = [
        ("Lahore", "Jazira"),
        ("Kasur", "Kasur"),
        ("Kasur", "Sahiwal"),
        ("Okara", "Kasur"),
        ("Okara", "Bakhar"),
        ("Okara", "Khosab"),
        ("Jhang", "Lahore"),
        ("Jhang", "Bakhar"),
        ("Khosab", "Okara"),
        ("Sahiwal", "Jhang")
    ]

    l = directed(t_city, pa, pair, city_names)

    for i in range(t_city):
        nei = l[i]
        neighbors = ", ".join(city_names[j] for j in nei)
        if neighbors:
            print(f"{city_names[i]}: {neighbors}")
        else:
            print(f"{city_names[i]}: x")
