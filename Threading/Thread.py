import threading
import time

def printNumbers(n):
    for i in range(n):
        time.sleep(2)
        print(f"Number ({i})")

def printLetter(string):
    for letter in string:
        time.sleep(3)
        print(f"Letter ({letter})")

# Create Threads
startTime = time.time()

t1 = threading.Thread(target=printNumbers, args=(5,))
t2 = threading.Thread(target=printLetter, args=("shivam",))

# Start Threads
t1.start()
t2.start()

# Wait for threads to complete
t1.join()
t2.join()

endTime = time.time()

timeTaken = endTime - startTime
print(f"\nTotal time taken: {timeTaken:.2f} seconds")
