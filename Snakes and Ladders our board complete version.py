"""NAMMA BOARD MATTHU RULES KUUDA!!!  FTR means the free turn rule. if a player gets a particular number of free turns he will have to start over again."""
from random import randint
Player1,Player2="Notwin","Notwin"
continue_till_both_wins=True
num1,num2,dice_roll_value1,dice_roll_value2,P1turn,P2turn,P1roll_num,P2roll_num=1,1,0,0,0,0,0,0
FTR = 3000000
snake = {51: 11, 56: 15, 62: 57, 92: 53, 98: 8}
ladder = {2: 38, 4: 14, 9: 31, 33: 85, 52: 88, 80: 99}
computer_password,human_password=1,1
print("Press Enter to begin!")
while (num1)<=100:
	if computer_password==1:
		P1turn=P1turn+1
		t=0
		while True:
			P1roll_num=P1roll_num+1
			f=input()
			print("Player 1 Rolls the die","("+str(P1roll_num)+")")
			dice_roll_value1=randint(1,6)
			print(dice_roll_value1)
			if num1+dice_roll_value1<=100:
				num1,e=num1+dice_roll_value1,num1+dice_roll_value1
				if num1 in snake:
					num1=snake[num1]
					print("snaked from",e,"to")
				if num1 in ladder:
					num1=ladder[num1]
					print("laddered from",e,"to")
				print("Num =",num1)
				if dice_roll_value1==6:
					print("Free turn!!")
					if num1==100:
						print("Player 1 won! in",P1turn,"turns")
						exit()
					t=t+1
					if t==FTR:
						print("But",FTR,"consecutive free turns are not counted!, You need to start from 1")
						num1=1
						print("Num =",num1)
						break
					continue
				else:
					break
			else:
				print("The number exceeds 100, so please try another number ")
				if dice_roll_value1==6:
					print("Free turn!!")
					if num1==100:
						print("Player 1 won! in",P1turn,"turns")
						exit()
					t=t+1
					if t==FTR:
						print("But",FTR,"consecutive free turns are not counted!, You need to start from 1")
						num1=1
						print("Num =",num1)
						break
					continue
				else:
					break
		if num1==100:
			Player1="win"
			print("Player 1 won! in",P1turn,"turns")
			computer_password=2
			if continue_till_both_wins==False:
				break
	if human_password==1:
		P2turn=P2turn+1
		tx=0
		while True:
			P2roll_num=P2roll_num+1
			f=input()
			print("Player 2 Rolls the die","("+str(P2roll_num)+")")
			dice_roll_value2=randint(1,6)
			print(dice_roll_value2)
			if num2+dice_roll_value2<=100:
				num2,e=num2+dice_roll_value2,num2+dice_roll_value2
				if num2 in snake:
					num2=snake[num2]
					print("snaked from",e,"to")
				if num2 in ladder:
					num2=ladder[num2]
					print("laddered from",e,"to")
				print("Num =",num2)
				if dice_roll_value2==6:
					print("Free turn!!")
					if num2==100:
						print("Player 2 won! in",P2turn,"turns")
						exit()
					tx=tx+1
					if tx==FTR:
						print("But",FTR,"consecutive free turns are not counted!, You need to start from 1")
						num2=1
						print("Num =",num2)
						break
					continue
				else:
					break
			else:
				print("The number exceeds 100, so please try another number ")
				if dice_roll_value2==6:
					print("Free turn!!")
					if num2==100:
						print("Player 2 won! in",P2turn,"turns")
						exit()
					tx=tx+1
					if tx==FTR:
						print("But",FTR,"consecutive free turns are not counted!, You need to start from 1")
						num2=1
						print("Num =",num2)
						break
					continue
				else:
					break
		if num2==100:
			print("Player 2 won! in",P2turn,"turns")
			Player2="win"
			human_password=2
			if continue_till_both_wins==False:
				break
	if Player1=="win" and Player2=="win":
		exit()