statement = ""
for i in range(1000):
    substatement = "(A"
    for i in range(100):
        substatement += f"&A{i}"
    substatement += ")&"
    statement += substatement
print(statement[:-1])
