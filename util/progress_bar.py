def render(total, current):
    full_count = int(current/total*10)
    if current == 0:
        half_count = 0
    else:
        half_count = round((current%(total/10))/(total/10))
    empty_count = 10 - (full_count+half_count)
    print(f"[{"█"*full_count}{"░"*half_count}{" "*empty_count}] ({round(current/total*100, 2)}%)")

if __name__ == "__main__":
    from time import sleep

    for i in range(331):
        render(331, i)
        sleep(0.05)