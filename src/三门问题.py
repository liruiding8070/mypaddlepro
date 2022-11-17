import random

def play(change):
    prize = random.randint(0, 2)
    guess = random.randint(0, 2)
    if guess == prize:
        return not change
    else:
        return change

def winRate(change, N):
    win = 0
    for i in range(N):
        if (play(change)):
            win += 1
    print("中奖率：{}".format(win / N))

N = 1000000
print("每次换门：")
winRate(True, N)
print("每次不换门：")
winRate(False, N)
