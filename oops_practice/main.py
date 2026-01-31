'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
# class Student:
   
     
#     def __init__(self,name,marks):
#         self.name=name 
#         self.marks=marks 
#     @staticmethod 
#     def hello():
#         print("Hello ")
    
#     def get_Average(self):
#         sum1=0
#         for val in self.marks:
#             sum1+=val 
#         return (sum1)/len(self.marks)
        
   
# s1=Student("Shivam",[80,85,90])
# print("Average marks=",s1.get_Average())
# print(Student.hello())
#Abstraction ----------------------------------------------------

# class Car:
#     def __init__(self):
#         self.acc=False 
#         self.clutch=False 
#         self.brk=False 
#     def start(self):
#         self.clutch=True 
#         self.acc=True 
#         print("Car Started")
# c1=Car()
# print("Clutch=",c1.clutch)
# c1.start()
# print("Clutch=",c1.clutch)

#Bank Practice 
class Account:
    def __init__(self,bal,acc):
        self. bal=bal 
        self.acc=acc
    def debit(self,amt):
        self.bal-=amt
    def credit(self,amt):
        self.bal+=amt
    def getBalance(self):
        return self.bal
        
        
    
    
    
acc1=Account(10000,12345)
print(acc1.bal)
acc1.debit(1000)
acc1.credit(2000)
print("Final balance=",acc1.getBalance())



