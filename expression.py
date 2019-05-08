import sys
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
special = ['+', '-', '=']


class Expression:

    def __init__(self, symbols=None, type=None, value=None, latex=None):
        self.__symbols = symbols
        self.__sym_length = len(symbols)
        self.type = type
        self.value = value
        self.latex = latex


    def resolve(self):

        if self.__sym_length < 1:
            print('Expression length lower than one!')
            sys.exit(-1)

        elif self.__sym_length == 1:
            bb = self.__symbols.pop(0)
            sym = bb.symbol
            value = sym if sym not in numbers else int(sym)

            return sym, value

        else:
            # 1. ------resolve fractions-----
            potential_div = [bb for bb in self.__symbols if bb.symbol in ['-', '=']]
            self.resolve_fractions(potential_div)

            self.__symbols = sorted(self.__symbols, key=lambda x: (x.xmin, x.ymin))
            self.mergeNumbers()
            self.__symbols = sorted(self.__symbols, key=lambda x: (x.xmin, x.ymin))
            # inital
            inital = True
            accumulator = 0
            opp = ''
            latex = ''
            for sym in self.__symbols:
                if inital:
                    if sym.type in ['CONST', 'FRACTION']:
                        accumulator += sym.value
                        latex += str(sym.value) if sym.latex is None else sym.latex
                        inital = False

                else:
                    if sym.type not in ['CONST', 'FRACTION']:
                        opp = '-' if sym.symbol == '=' else sym.symbol
                        latex += opp

                    else:
                        latex += str(sym.value) if sym.latex is None else sym.latex
                        accumulator = self.doOpertaion(accumulator, opp, sym.value)
                        opp = ''

        return latex, accumulator


    def mergeNumbers(self):
        copy_sym = [x for x in self.__symbols]
        buffer = []
        for sym in copy_sym:

            if sym.symbol == 'FRACTION': continue

            if sym.symbol in numbers or sym.type == 'CONST':
                buffer.append(sym)
                self.__symbols.remove(sym)

            else:
                if len(buffer) != 0:
                    string = ''
                    for bb in buffer: string += bb.symbol
                    first = buffer[0]
                    first.symbol = string
                    first.value = int(string)
                    first.type = 'CONST'
                    first.update_borders(buffer)
                    self.__symbols.append(first)
                    buffer = []

        if len(buffer) != 0:
            string = ''
            for bb in buffer: string += bb.symbol
            first = buffer[0]
            first.symbol = string
            first.value = int(string)
            first.type = 'CONST'
            first.update_borders(buffer)
            self.__symbols.append(first)
            buffer = []



    def doOpertaion(self, num1, operation, num2):
        rez = 0
        if operation == '+':
            rez = num1 + num2
        elif operation == '-':
            rez = num1 - num2

        return rez

    def resolve_fractions(self, potential_div):
        potential_div = sorted(potential_div, key=lambda x: (-(x.xmax - x.xmin), x.xmin))

        while len(potential_div) > 0:
            temp_symbol = potential_div.pop(0)
            if temp_symbol not in self.__symbols: continue

            above_list = []
            under_list = []
            copy_symbols = [x for x in self.__symbols]

            for symbol in copy_symbols:

                if symbol.id == temp_symbol.id:
                    continue

                elif symbol.is_above(temp_symbol):
                    above_list.append(symbol)

                elif symbol.is_under(temp_symbol):
                    under_list.append(symbol)

                else: continue
                self.__symbols.remove(symbol)

            len_up = len(above_list)
            len_down = len(under_list)

            if max(len_up, len_down) == 0:
                #minus
                temp_symbol.value = '-'
                temp_symbol.type = 'OPT'
                temp_symbol.latex = '-'
                continue

            if min(len_up, len_down) == 0:
                print('EROoR')
                sys.exit(-1)

            latex1, value1 = Expression(symbols=above_list).resolve()
            latex2, value2 = Expression(symbols=under_list).resolve()

            # Update fraction bb
            temp_symbol.value = value1/value2
            temp_symbol.type = 'FRACTION'
            temp_symbol.latex = r'\frac{' + latex1 + '}{' + latex2 + '}'
            temp_symbol.update_borders(above_list + under_list + [temp_symbol])
