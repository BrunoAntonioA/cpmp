import numpy as np
import random
import copy
from queue import PriorityQueue


# clase que define los stacks de containers
class ContainerStack:
    def __init__(self, group_value, stack):
        self.group_value = group_value
        self.stack = stack

    # método para obtener group value de una lista o de un stack
    def get_group_value(self):
        if len(self.stack) <= 1:
            self.group_value = 0
            return self.group_value
        for x in range(1, len(self.stack), 1):
            if self.stack[x] > self.stack[x - 1]:
                self.group_value = (len(self.stack) - x)
                return self.group_value
        self.group_value = 0
        return self.group_value


# final de clase ContainerStack

# clase para definir un estado; es la clase contenedora de los stacks de containers
class Yard:
    def __init__(self, container_array, max_height, min_cost):
        self.container_array = container_array
        self.max_height = max_height
        self.min_cost = min_cost

    # Define el min cost de un yard
    def get_min_cost(self):
        min_cost = 0
        for x in self.container_array:
            min_cost = min_cost + x.group_value
        self.min_cost = min_cost

        # método para mover un container de la columna c1 a la c2, recibe el estado

    # Metodo que genera un movimiento dentro del Yard, actualiza group values y el min cost
    # En caso de no poder realizarse el movimiento, retorna 0, en caso de que si se realizo
    # Entrega el mismo nodo Yard
    def movement(self, c1, c2):

        if len(self.container_array[c1].stack) != 0 and (len(self.container_array[c2].stack) < self.max_height):
            # hago un pop a c1 y con el mismo elemento que borré hago un append a c2
            self.container_array[c2].stack.append(self.container_array[c1].stack.pop())
        else:
            print("No se puede realizar este movimiento. La columna ", c1 + 1, " no tiene elementos o la columna ",c2 + 2, " está llena")
            return 0
        self.container_array[c1].get_group_value()
        self.container_array[c2].get_group_value()
        self.get_min_cost()
        return self

    # método que evalua un estado si es que está ordenado o no
    def eval_state(self):
        for x in range(len(self.container_array)):
            if self.container_array[x].group_value != 0:
                return 0
        return 1

    # método para imprimir un estado
    def print_yard(self):
        for y in range(self.max_height - 1, -1, -1):
            for x in range(len(self.container_array)):
                if y > len(self.container_array[x].stack) - 1:
                    print(0, end='   ')
                else:
                    print(self.container_array[x].stack[y], end='   ')
            print(" ")
            print(" ")

    # método que entrega un conjunto de las acciones posibles en un estado
    def get_posible_actions(self):
        actions = []
        for x1 in range(len(self.container_array)):
            # columna no está vacía?
            if self.container_array[x1].stack:
                for x2 in range(len(self.container_array)):
                    # columna 2 tiene espacio?
                    if x1 != x2:
                        if self.max_height > len(self.container_array[x2].stack):
                            actions.append((x1, x2))
        return actions

    # Metodo que transforma un estado desde la clase Yard a un array
    def transform_to_array(self):
        array = np.array([])
        for x in range(len(self.container_array)):
            for y in range(self.max_height):
                if y < len(self.container_array[x].stack):
                    array = np.append(array, self.container_array[x].stack[y])
                else:
                    array = np.append(array, 0)
        return array.astype(int)

    # Metodo que entrega un arreglo con la cantidad
    # De container desordenados de cada columna
    def group_values_array(self):
        gv_a = np.array([])
        for x in self.container_array:
            gv_a = np.append(gv_a, x.group_value)
        return gv_a.astype(int)

    # Metodo que entrega el valor del mayor container
    def mayor_container(self):
        mayor = 0
        for x in self.container_array:
            for y in x.stack:
                if y > mayor:
                    mayor = y
        return mayor

    # Metodo que obtiene las diferencias del primer container
    # De cada columna con el mayor de todos los container de todas las columnas
    # Se resuelve con un array, porque se ocupa sobre el estado convertido
    # En un array ya normalizado

    def get_base_differences(self):
        mayor = self.mayor_container()
        differences = np.array([])
        for x in self.container_array:
            if len(x.stack) == 0:
                differences = np.append(differences, mayor)
            else:
                difference = mayor - x.stack[0]
                differences = np.append(differences, difference)
        return differences.astype(int)

    def get_base_differences_from_normalize(self, array):
        differences = np.array([])
        mayor = np.amax(array)
        f = self.max_height
        for x in range(0, len(array), f):
            dif = mayor - array[x]
            differences = np.append(differences, dif)
        return differences.astype(int)

    # Metodo que obtiene las diferencias del ultimo container
    # De cada columna con el mayor de todos los container de todas las columnas
    # Se resuelve con un array, porque se ocupa sobre el estado convertido
    # En un array ya normalizado
    def get_top_differences(self):
        mayor = self.mayor_container()
        differences = np.array([])
        for x in self.container_array:
            if len(x.stack) == 0:
                differences = np.append(differences, mayor)
            else:
                difference = mayor - x.stack[len(x.stack)-1]
                differences = np.append(differences, difference)
        return differences.astype(int)

    def get_top_differences_from_normalize(self, array):
        differences = np.array([])
        mayor = np.amax(array)
        f = self.max_height
        for x in range(0, len(array), f):
            if array[x] == 0:
                dif = mayor
                differences = np.append(differences, dif)
                continue
            for i in range(x, x+f, 1):
                if array[i] == 0:
                    dif = mayor - array[i-1]
                    differences = np.append(differences, dif)
                    break
                if i == x + f - 1:
                    dif = mayor - array[x+f-1]
                    differences = np.append(differences, dif)
        return differences.astype(int)

    # Metodo de ayuda para el array_pilas_necesarias
    def pilas_necesarias(self, stack, i):
        aux = []
        count = 0
        for x in range(len(stack)-1, i, -1):
            count = count + 1
            flag = 0
            if len(aux) == 0:
                aux.append(stack[x])
                continue
            for z in range(len(aux)):
                if stack[x] <= aux[z]:
                    aux[z] = stack[x]
                    flag = 1
                    break
            if flag == 0:
                aux.append(stack[x])
        return len(aux)

    # Metodo que cuenta las cantidades de pilas necesarias para desarmar
    # cada pila. Retorna un array con esta cantidad por cada columna
    def array_pilas_necesarias(self):
        pilas_arr = np.array([])
        for x in self.container_array:
            if len(x.stack) == 0 or len(x.stack) == 1:
                pilas_arr = np.append(pilas_arr, 0)
                continue
            for y in range(len(x.stack)-1):
                if x.stack[y] < x.stack[y+1]:
                    pilas_arr = np.append(pilas_arr, self.pilas_necesarias(x.stack, y))
                    break
                if y == len(x.stack)-2:
                    pilas_arr = np.append(pilas_arr, 0)
        return pilas_arr.astype(int)

    #metodo que organiza las columnas de un estado
    def organizar_por_columnas(self):
        a_gv = self.group_values_array()
        a_gv_order = np.sort(a_gv)
        count = 0
        for x in a_gv_order:
            for y in np.where(a_gv == x):
                aux = a_gv[count]
                a_gv[count] = a_gv[y]
                a_gv[y] = aux
                count = count + 1
        return 1

    # termina la clase yard


# Clase para un nodpo
class Node:

    def __init__(self, data):
        self.data = data
        self.parent = None
        self.children = []

    def cost(self):
        count = 0
        s = self
        while s.parent:
            count = count + 1
            s = s.parent
        if self.parent is None:
            return 0
        return count

# termina la clase Node


# Clase que hereda la clase Node, aparte tiene los metodos para el CPMP
class CPMP_Node(Node):
    def __init__(self, data):
        super().__init__(data)
        self.steps = []

    # agrega los posibles estados como hijos de un estado
    def contar_profundidad(self):
        count = 0
        cop = copy.deepcopy(self)
        while cop.parent:
            count = count + 1
            cop = cop.parent
        return count

    # Metodo que genera los posibles estados a partir de un nodo
    def add_childen(self):
        count = 0
        for x in self.data.get_posible_actions():
            nc = copy.deepcopy(self.data)
            nc = nc.movement(x[0], x[1])
            nc_node = CPMP_Node(nc)
            # pasos del padre para llegar a el nodo
            nc_node.steps.append((x[0], x[1]))
            nc_node.parent = self
            self.children.append(nc_node)
            count = count + 1
        return count

    # Genera el valor esperado por la red en base al movimiento que se debe realizar
    def generar_y(self):
        if self.parent is None:
            return -1
        tupla = self.steps[0]
        c1 = tupla[0]
        c2 = tupla[1]
        count = 0
        column = len(self.data.container_array)
        for i in range(column):
            for j in range(column):
                if j == i:
                    continue
                if (c1 == i) and (c2 == j):
                    return count
                count = count + 1

    # Este metodo genera un array con el estado del problema y sus atributos, a partir de un nodo CPMP node
    def generate_data(self):
        data = []
        s = self
        while s.parent:
            array = normalize_array(s.parent.data.transform_to_array())
            cp = copy.deepcopy(array)
            array = np.append(array, s.parent.data.group_values_array())
            array = np.append(array, s.parent.data.get_base_differences_from_normalize(cp))
            array = np.append(array, s.parent.data.get_top_differences_from_normalize(cp))
            array = np.append(array, s.parent.data.array_pilas_necesarias())
            array = np.append(array, s.parent.generar_y())
            data.append((array, s.generar_y()))
            s = s.parent
        return data

    def __eq__(self, other):
        if not isinstance(other, CPMP_Node):
            return 0
        if (self.data.max_height != other.data.max_height) | (
                len(self.data.container_array) != len(other.data.container_array)):
            return 0
        for x in range(len(self.data.container_array)):
            if len(self.data.container_array[x].stack) != len(other.data.container_array[x].stack):
                return 0
            for y in range(len(self.data.container_array[x].stack)):
                if self.data.container_array[x].stack[y] != other.data.container_array[x].stack[y]:
                    return 0
        return 1

    def __hash__(self):
        hash_number = 0
        for x in range(len(self.data.container_array)):
            for y in range(len(self.data.container_array[x].stack)):
                hash_number = hash_number + x * self.data.container_array[x].stack[y] - \
                              self.data.container_array[x].stack[y]
        return hash(hash_number + 2943)

# termina la clase CPMP_Node


# metodo del arbol de busqueda optima para resolver un estado
def dlts_lds(s):
    q = PriorityQueue(0)
    estados_agregados = set()
    count = 0
    q.put((s.cost() + s.data.min_cost, count, s))
    while not q.empty():
        s = q.get()
        s = s[2]
        if s.data.eval_state() == 1:
            return s
        s.add_childen()
        for s_child in s.children:
            if not(s_child in estados_agregados):
                hlb = s_child.cost() + s_child.data.min_cost
                estados_agregados.add(s_child)
                q.put((hlb, count,  s_child))
                count += 1


# método para inicializar un estado con 'x' columnas, 'y' filas con numeros aleatorios
# y 'discount' filas rellenadas con '0'

def init(x, y, discount):
    s0 = Yard([], y, 0)
    for j in range(x):
        stack = ContainerStack(0, [])
        for k in range(y - discount):
            stack.stack.append(random.randrange(1, 99, 1))
        stack.get_group_value()
        s0.container_array.append(stack)
        s0.min_cost = s0.min_cost + s0.container_array[j].group_value

    return s0


# metodo para normalizar array, los de los container cambian, basandose en su orden
# de menor a mayor
def normalize_array(array):
    order_array = []
    cop = copy.deepcopy(array)
    cop.sort()
    for x in cop:
        if x not in order_array:
            order_array.append(x)
    for z in range(len(array)):
        array[z] = order_array.index(array[z])
    return array.astype(int)


# Se generan estados aleatorios con sus atributos, los que se guardan en un .txt
# Se diferencia de cada array por una ','
# n1 representa la cantidad de estados generados para el entrenamiento
# n2 representa la cantiad de estados generados para el testeo
# c cantidad de columnas de las instancias a generar
# f cantidad de filas de las instancias a generar
# ds cantidad de filas superiores en blanco que se desean sobreponer
def generate_data_set(n1, n2, c, f, ds):
    d = open("data/states/train.txt", "a")
    l = open("data/states/trainlabel.txt", "a")
    inputs = []
    outputs = []
    for x in range(n1):
        s = init(c, f, ds)
        if s.eval_state():
            continue
        node = CPMP_Node(s)
        s = dlts_lds(node)
        data = s.generate_data()
        for z in data:
            y = z[0]
            inputs.append(y)
            outputs.append(z[1])
    for x in inputs:
        for y in range(len(x)):
            d.write(str(x[y]))
        d.write(",")
    for x in outputs:
        l.write(str(x))
        l.write(",")
    d.close()
    l.close()

    d = open("data/states/test.txt", "a")
    l = open("data/states/testlabel.txt", "a")
    inputs = []
    outputs = []
    for x in range(n2):
        s = init(c, f, ds)
        if s.eval_state():
            continue
        node = CPMP_Node(s)
        s = dlts_lds(node)
        data = s.generate_data()
        for z in data:
            y = z[0]
            inputs.append(y)
            outputs.append(z[1])
    for x in inputs:
        for z in x:
            d.write("-")
            d.write(str(z))
        #for y in range(len(x)):
        #    d.write(str(x[y]))
        d.write(",")
    for x in outputs:
        l.write(str(x))
        l.write(",")
    d.close()
    l.close()


generate_data_set(10, 10, 3, 3, 1)

'''
    El vector generado tiene el siguiente formato:
    [0-N] El estado generado transformado en un array, donde N = (Filas * Columnas) - 1
    [N-N+Columnas-1] Por cada posicion de este rango, es la cantidad de container desordenados de su correspondiente columna
     

'''