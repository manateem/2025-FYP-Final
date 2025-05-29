def render(total, current):
    full_count = int((current+1)/total*10)
    if current == 0 or full_count == 10:
        half_count = 0
    else:
        half_count = round(((current+1)%(total/10))/(total/10))
    empty_count = 10 - (full_count+half_count)
    print(f"[{bytes((219,)).decode('cp437')*full_count}{"░"*half_count}{" "*empty_count}] ({round((current+1)/total*100, 2)}%)")

if __name__ == "__main__":
    from time import sleep
    import os

    for i in range(331):
        os.system('clear')
        render(331, i)
        sleep(0.05)