# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 04:28:03 2020

@author: SAKSHAM
"""

while True:
    balance = 10000
    print(" ATM ")
    print("""
          1) Balance
          2) Withddraw
          3) Deposit
          4) Quit
         """)
    try:
        Option = int(input("Enter Option: "))
    except Exception as e:
        print("Error:", e)
        print(" Enter 1,2,3 or 4 only")
        continue
    if Option == 1:
        print("Balance $", balance)
    if Option == 2:
        print("Balance $", balance)
        withdraw=float(input("Enter withdraw amounnt $"))
        if withdraw>0:
            forwardbalance=(balance-withdraw)
            print("forward Balance $",forwardbalance)
        elif withdraw>balance:
            print("No funds in account")
        else:
            print("None withdraw made")
    if Option == 3:
        print("Balance $", balance)
        Deposit=float(input("Enter deposit amount $"))
        if Deposit>0:
            forwardbalance=(balance + Deposit)
            print("forwardbalance $",forwardbalance)
        else:
            print("None deposit made")
    if Option == 4:
        exit(Quit)