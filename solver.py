
def cross(X, Y ) :
    return [x + y for x in X for y in Y]

digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits

squares = cross(rows, cols)

unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')])

#print(squares)
#print(unitlist)

units = dict((s, [u for u in unitlist if s in u])
                for s in squares)
peers = dict((s, set(sum(units[s], []))- set([s]))
                for s in squares)


with open('example3.txt', 'r') as F:
    gridText = F.read()

gridText = gridText.replace('\n', '')
# print(gridText)

def showGrid(values):

    width = 1 +max(len(values[s]) for s in squares)
    tableLine = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        for c in cols:
            if c in ['4', '7']:
                print('|', end='')
            print(f"{values[r+c]:^{width}}", end='')
        
        if r in ['C', 'F']:
            print()
            print(tableLine,end='')
        print()


def eliminate(values, sq, digit) :

    if digit not in values[sq]:
        return values
    
    values[sq] = values[sq].replace(digit, '')

    if len(values[sq]) == 0:
        return False
    elif len(values[sq]) == 1:
        d = values[sq]
        if not all(eliminate(values, s, d) for s in peers[sq]):
            return False
    
    for u in units[sq]:
        dplaces = [s for s in u if digit in values[s]]

        if len(dplaces) == 0:
            return False
        elif len(dplaces) == 1:
            if not assign(values, dplaces[0], digit):
                return False

    return values

def assign(values, sq, digit) :
    otherDigits = values[sq].replace(digit,'')
    if all(eliminate(values, sq, d) for d in otherDigits) :
        return values
    else:
        return False


def grid_values(grid) :
    numbers = [num for num in grid]
    assert 81 == len(grid)
    return dict(zip(squares, numbers))

def parse_grid(grid) :

    values = dict((s,digits) for s in squares)
    givenNumbers = grid_values(gridText)
    
    # for s, d in givenNumbers.items():
    #     print('s:{} and possible values:{}'.format(s,d))      
    
    for sq, d in givenNumbers.items():
        if d in digits and not assign(values, sq, d):
            return False
    
    return values


def some(sequences) :
    for seq in sequences:
        if seq:
            return seq
    
    return False


def search(values):

    if values is False:
        return False
    if  all(len(values[sq]) == 1 for sq in squares):
        return values
    
    minNumbers, respectiveSq = min((len(values[sq]), sq ) for sq in squares if len(values[sq])>1 )

    return some(search(assign(values.copy(), respectiveSq, d))
                for d in values[respectiveSq])
    

initialGrid = parse_grid(gridText)
print("initial grid after entry of data:")
showGrid(initialGrid)

finalAnswer = search(initialGrid)
print("\n\nfinal grid after solving:")
showGrid(finalAnswer)

# print(squares)

# for s, l in units.items():
#     print('s:{} and units:{}'.format(s,l))

# print('-------------')

# for s, d in initialGrid.items():
#     print('s:{} and possible values:{}'.format(s,d))

