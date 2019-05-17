import sys

# ex: python get_times.py exp1/experiment1.txt

if __name__ == "__main__":
    
    with open(sys.argv[1]) as f:
        last = 0
        for l in f:
            curr = float(l.split('T:')[1].split('\t')[0])
            print(curr, end='  ')
            if last:
                print(curr - last)
            else:
                print('')
            last = curr
