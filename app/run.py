from agents import agent

while True:
    query = input("\nAsk: ")

    if query.lower() == "exit":
        print("Exiting...")
        break

    result = agent(query)
    print("\nResult:\n", result)