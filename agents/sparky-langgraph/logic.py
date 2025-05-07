# ask user if he ran the application already
    # if he ran already, ask him if he wants to share the code.
        # if he wants to share the code, ask him to load the code in the format and the folder structure and route to code analyzer agent.
        # if he doesn't want to share the code, ask him to load the UI pdfs in the format and the folder structure and route to UI analyzers.
    # if he didn't run ask if he wants to share the code.
        # if he wants to share the code, ask him to load the code in the format and the folder structure and route to code description setter.
            #  After the description setting, route the code to the code analyzer agent and share the outputs (suggestions) with the user.
            # After that, ask him to run the description setted code and come back again.
        # if he doesn't want to share the code, ask him to run the application and come back.

is_Ran = input("Hello and welcome to te Spark Companion. Have you ran the application before? (yes/no): ").strip().lower()
if is_Ran == "yes":
    # Ask if the user wants to share the code
    share_code = input("Do you want to share the code? (yes/no): ").strip().lower()
    if share_code == "yes":
        # Route to code analyzer agent
        print("Routing to Code Analyzer Graph")
    else:
        # Route to UI analyzers
        print("Please load the UI PDFs without changing their names after downloading and type yes.")
        if input("Have you loaded the UI PDFs? (yes/no): ").strip().lower() == "yes":
            print("Routing to UI Analyzer Graph")
else:
    # Ask if the user wants to share the code
    share_code = input("Do you want to share the code? (yes/no): ").strip().lower()
    if share_code == "yes":
        # Route to code description setter
        print("Routing to Code Description Setter Graph")
    else:
        # Route to run the application
        print("Please go run the app first and come back.")
