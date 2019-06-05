import sys, math
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
special = ['+', '-', '=']
trig_and_log =['sin', 'cos', 'tan', 'log']


"""
-problem skaliranja slike
-skeletnizacija - debljina
-presjek
"""

class Expression:

    def __init__(self, symbols=None, type=None, value=None, latex=None):
        self.__symbols = symbols
        self.__sym_length = len(symbols)
        self.type = type
        self.value = value
        self.latex = latex

    def resolve(self):

        if self.__sym_length < 1:
            print('[ERROR] Expression length lower than one!')
            sys.exit(-1)

        elif self.__sym_length == 1:
            self.resolve_initial_constants()
            element = self.__symbols.pop(0)

            if element.symbol in numbers:
                return element.symbol, int(element.symbol)
            else:
                return element.latex, element.value

        else:
            self.__symbols = sorted(self.__symbols, key=lambda x: (x.xmin, x.ymin))
            # 2. Resolve fractions
            self.resolve_fractions()

            # 1. Resolve e and pi constants
            self.resolve_initial_constants()
            # ONE LINE EVERYTHING

            # 4. Resolve sin, cos, tan, log
            self.resolve_trigonometry_and_log()

            # 5. Resolve (, )
            while self.find_and_resolve_parentheses_unit([x for x in self.__symbols]): pass

            # 6. Resolve sqrt

            # 3. Create numbers
            self.mergeNumbers()

            # 7. Do multiply
            self.resolve_multiplication()
            # 8. Resolve plus and minus
            return self.resolve_plus_and_minus()

    def resolve_initial_constants(self):
        for sym in self.__symbols:
            if sym.symbol == 'e':
                sym.type = 'CONST'
                sym.value = math.e
                sym.latex = 'e'

            elif sym.symbol == 'pi':
                sym.type = 'CONST'
                sym.value = math.pi
                sym.latex = r'\pi'

            elif sym.symbol == '+':
                sym.type = 'OPERATION'
                sym.value = None
                sym.latex = '+'

            elif sym.symbol == 'times':
                sym.type = 'OPERATION'
                sym.value = None
                sym.latex = r'\times'


    def resolve_fractions(self):
        potential_div = [bb for bb in self.__symbols if bb.symbol == '-']
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
                temp_symbol.value = None
                temp_symbol.type = 'OPERATION'
                temp_symbol.latex = '-'
                continue

            if min(len_up, len_down) == 0:
                print('EROoR')
                sys.exit(-1)

            latex1, value1 = Expression(symbols=above_list).resolve()
            latex2, value2 = Expression(symbols=under_list).resolve()

            # Update fraction bb
            temp_symbol.value = value1/value2
            temp_symbol.symbol = None
            temp_symbol.type = 'CONST'
            temp_symbol.latex = r'\frac{' + latex1 + '}{' + latex2 + '}'
            temp_symbol.update_borders(above_list + under_list + [temp_symbol])

    def resolve_multiplication(self):
        self.__symbols = sorted(self.__symbols, key=lambda x: (x.xmin, x.ymin))

        copy = [x for x in self.__symbols]
        last = None
        last_type = ''
        to_resolve = []
        mylist = set()

        for sym in copy:

            if sym.type == 'CONST' and last_type == 'CONST':
                mylist.add(sym)
                mylist.add(last)

            else:
                if len(mylist) > 1:
                    to_resolve.append(list(mylist))

                mylist = set()

            # Mark
            last = sym
            last_type = sym.type

        if len(mylist) > 1: to_resolve.append(list(mylist))

        for elements in to_resolve:
            elements = sorted(elements, key=lambda x: (x.xmin, x.ymin))
            first = elements[0]
            acc = first.value
            latex = first.latex

            for el in elements[1:]:
                self.__symbols.remove(el)
                acc *= el.value
                latex += el.latex

            first.value = acc
            first.latex = latex
            first.symbol = latex
            first.type = 'CONST'
            first.update_borders(elements)

        self.__symbols = sorted(self.__symbols, key=lambda x: (x.xmin, x.ymin))
        while True:
            # Find first index to resolve
            index_to_resolve = next((i for i, sym in enumerate(self.__symbols) if sym.symbol == 'times'), None)

            # End of trigonometry resolve
            if index_to_resolve is None: return

            toResolve = [self.__symbols[i] for i in [index_to_resolve -1, index_to_resolve, index_to_resolve + 1]]
            first, opp, second = toResolve[0], toResolve[1], toResolve[2]

            first.value = self.doOpertaion(first.value, opp.symbol, second.value)
            first.latex += opp.latex + second.latex
            first.symbol = first.latex
            first.type = 'CONST'
            first.update_borders([first, opp, second])
            self.__symbols.remove(opp)
            self.__symbols.remove(second)

        #self.__symbols = sorted(self.__symbols, key=lambda x: (x.xmin, x.ymin))
        #a = [i for i, bb in enumerate(self.__symbols) if bb.symbol == 'times']



    def resolve_plus_and_minus(self):
        self.__symbols = sorted(self.__symbols, key=lambda x: (x.xmin, x.ymin))
        symbol_length = len(self.__symbols)
        accumulator = 0
        latex = ''

        # Check first element
        first_element = self.__symbols[0]
        if first_element.type != 'CONST':
            print('[ERROR] Expression can not start with OPERATION')
        else:
            accumulator += first_element.value
            latex += first_element.latex

        if symbol_length == 2:
            print('ERROR: Need constatn')
            sys.exit(-1)

        else:
            for index in range(1, symbol_length - 1):
                operation = self.__symbols[index].symbol
                constant = self.__symbols[index + 1]
                accumulator = self.doOpertaion(accumulator, operation, constant.value)
                latex += operation + constant.latex

        return latex, accumulator


    def find_and_resolve_parentheses_unit(self, symbols, hook_to=None):
        load_symbols = False
        buffer = []
        counter = 0

        for sym in symbols:

            if load_symbols:
                buffer.append(sym)
                self.__symbols.remove(sym)

                if sym.symbol == '(': counter += 1

                if sym.symbol == ')':
                    if counter == 0:
                        # Recursive (without parentheses)
                        latex, value = Expression(symbols=buffer[1:-1]).resolve()

                        # Update
                        hook_to.value = value
                        hook_to.type = 'CONST'
                        hook_to.latex = ''
                        if hook_to.symbol != '(':
                            hook_to.latex += '\\' + hook_to.symbol
                            hook_to.value = self.doOpertaion(hook_to.value, operation=hook_to.symbol)
                        hook_to.latex += '(' + latex + ')'
                        hook_to.symbol = None
                        hook_to.update_borders(symbols + [hook_to])
                        return True

                    else: counter -= 1

            elif sym.symbol == '(':
                if hook_to is not None: self.__symbols.remove(sym)
                else: hook_to = sym

                buffer.append(sym)
                load_symbols = True

        return False

    def resolve_trigonometry_and_log(self):
        self.__symbols = sorted(self.__symbols, key=lambda x: (x.xmin, x.ymin))

        while True:
            # Find first index to resolve
            index_to_resolve = next((i for i, sym in enumerate(self.__symbols) if sym.symbol in trig_and_log), None)

            # End of trigonometry resolve
            if index_to_resolve is None: return

            # Resolve
            sym_to_resolve = self.__symbols[index_to_resolve]
            next_symbols = [x for x in self.__symbols[index_to_resolve+1:]]

            if next_symbols[0].symbol != '(':
                print('[ERROR] {} should start with "("'.format(sym_to_resolve.symbol))
                sys.exit(-1)

            success = self.find_and_resolve_parentheses_unit(symbols=next_symbols, hook_to=sym_to_resolve)

            if not success:
                print('[ERROR] {} end of  ")"'.format(sym_to_resolve.symbol))
                sys.exit(-1)


    def merge(self, buffer):
        number_str = ''.join([bb.symbol for bb in buffer])
        first = buffer[0]
        first.symbol = number_str
        first.latex = number_str
        first.value = int(number_str)
        first.type = 'CONST'
        first.update_borders(buffer)
        self.__symbols.append(first)

    def merge2(self,  buffer2):
        """
        sve je u jednoj liniji
        :return:
        """
        buffer2 = sorted(buffer2, key=lambda x: (x.xmin, x.ymin))
        buffer = []

        for sym in buffer2:

            if sym.type == 'SYMBOL':
                buffer.append(sym)
                self.__symbols.remove(sym)

            elif sym.type in ['CONST', 'OPERATION']:
                if len(buffer) > 0:
                    self.merge(buffer)
                    buffer = []

            else:
                print('[ERROR] In "mergeNumbers()" symbol type unrecognized. Got: {}'.format(str(sym.type)))
                sys.exit(-1)

        # Maybe buffer is not empty
        if len(buffer) > 0: self.merge(buffer)


    def mergeNumbers(self):
        """
        sve je u jednoj liniji
        :return:
        """
        self.__symbols = sorted(self.__symbols, key=lambda x: (x.xmin, x.ymin))
        copy_symbols = [x for x in self.__symbols]
        buffer = []

        last_c = None
        buffer_center = []
        buffer_up = []

        # is upper buffer
        for sym in copy_symbols:

            if len(buffer_center) == 0:
                if sym.type != 'OPERATION':
                    buffer_center.append(sym)


            else:
                last_el = buffer_center[-1]

                if len(buffer_up) == 0:
                    if sym.is_above_and_right(last_el):
                        buffer_up.append(sym)

                    else:
                        if sym.type != 'OPERATION':
                            buffer_center.append(sym)
                        else:
                            self.merge2(buffer_center)
                            buffer_center = []
                else:
                    if sym.is_above_and_right(last_el):
                        buffer_up.append(sym)

                    else:
                        for s in buffer_center: self.__symbols.remove(s)
                        for s in buffer_up: self.__symbols.remove(s)

                        latex1, value1 = Expression(symbols=buffer_center).resolve()
                        latex2, value2 = Expression(symbols=buffer_up).resolve()

                        # Update fraction bb
                        last_el.value = math.pow(value1, value2)
                        last_el.symbol = None
                        last_el.type = 'CONST'
                        last_el.latex = latex1 + '^{' + latex2 + '}'
                        last_el.update_borders(buffer_center + buffer_up + [last_el])
                        self.__symbols.append(last_el)

                        buffer_center = [] if sym.type == 'OPERATION' else [sym]
                        buffer_up = []

        len1 = len(buffer_center)
        len2 = len(buffer_up)

        if len2 == 0:
            if len1 == 0: return
            else:
                self.merge2(buffer_center)
        else:
            if len1 == 0:
                print('error')
            else:
                last_el = buffer_center[-1]
                for s in buffer_center: self.__symbols.remove(s)
                for s in buffer_up: self.__symbols.remove(s)

                latex1, value1 = Expression(symbols=buffer_center).resolve()
                latex2, value2 = Expression(symbols=buffer_up).resolve()

                # Update fraction bb
                last_el.value = math.pow(value1, value2)
                last_el.symbol = None
                last_el.type = 'CONST'
                last_el.latex = latex1 + '^{' + latex2 + '}'
                last_el.update_borders(buffer_center + buffer_up + [last_el])
                self.__symbols.append(last_el)

    def doOpertaion(self, num1, operation, num2=None):
        rez = 0
        if operation == '+':
            rez = num1 + num2
        elif operation == 'times':
            rez = num1 * num2
        elif operation == '-':
            rez = num1 - num2
        elif operation == 'sin':
            rez = math.sin(num1)
        elif operation == 'cos':
            rez = math.cos(num1)
        elif operation == 'tan':
            rez = math.tan(num1)
        elif operation == 'log':
            rez = math.log(num1)

        return rez


