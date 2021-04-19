from sys import argv


def main():
    if len(argv) < 3:
        print(f"usage: {argv[0]} <file>")
    filepath = argv[1]
    with open(filepath) as fp:
        cnt = 0
        for line in fp:
            print("line {} contents {}".format(cnt, line))
            cnt += 1


if __name__ == '__main__':
    main()