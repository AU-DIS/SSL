#should create a string file and be able to push it
def main():
    print("hell")
    #create a string file
    string_file = open("string_file.txt", "w")
    string_file.write("This is a string file")
    string_file.close()
    balanced_accuracies = []
    accuracies = []
    recalls = []
    precisions = []
    f1s = []
    for i in range(0, 10):
        balanced_accuracies.append(i)
        accuracies.append(i)
        recalls.append(i)
        precisions.append(i)
        f1s.append(i)
    f = open('balanced_accuracy.txt', 'a+')
    f.write(str(balanced_accuracies))
    f = open('accuracy.txt', 'a+')
    f.write(str(accuracies))
    f = open('recall.txt', 'a+')
    f.write(str(recalls))
    f = open('precision.txt', 'a+')
    f.write(str(precisions))
    f = open('f1.txt', 'a')
    f.write(str(f1s))

main()