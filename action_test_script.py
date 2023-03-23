#should create a string file and be able to push it
def main():
    print("hell")
    #create a string file
    string_file = open("string_file.txt", "w")
    string_file.write("This is a string file")
    string_file.close()
main()