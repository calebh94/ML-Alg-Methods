#!/usr/bin/python
# CSE6140 HW2
# Implemented by Caleb Harris on 9/30/2020
import time
import sys
import os
# import heapq
import numpy as np
from queue import PriorityQueue
import heapq
import datetime


class Node():
    def __init__(self, index):
        self.i = index
        self.neighbors = []

    def add_neighbors(self, neigh_list):
        self.neighbors.append(neigh_list)

    def get_index(self):
        return self.i


class Graph():
    def __init__(self):
        self.V = np.array([])
        self.E = np.array([])
        self.nodes = 0
        self.edges = 0

    def get_size(self):
        return (self.nodes, self.edges)

    def allocate(self, num_nodes):
        # self.V = np.empty((num_nodes), dtype=np.int8)
        self.E = np.zeros((num_nodes, num_nodes), dtype=np.int8)
        self.V = []
        for i in range(0, num_nodes):
            node = Node(i)
            self.V.append(node)

    def remove_edge(self, node1, node2):
        self.E[node1, node2] = 0
        self.E[node2, node1] = 0
        self.V[node1].neighbors.remove(node2)
        self.V[node2].neighbors.remove(node1)

    def insert_edge(self, node1, node2, weight):
        # Check node in list
        if node1 >= len(self.V) or node2 >= len(self.V):
            raise ValueError("The input edge ({}, {}) has a node index outside the size, "
                             "{},  of the graph!".format(node1, node2, len(self.V)))
        if self.E[node1, node2] == 0:
            self.V[node1].add_neighbors(node2)
            self.V[node2].add_neighbors(node1)
            self.E[node1, node2] = weight
            self.E[node2, node1] = weight
        else:
            #TODO: update to have multiple edge values for different "edges"
            if weight < self.E[node1, node2]:
                self.E[node1, node2] = weight
                self.E[node2, node1] = weight

        self.edges = self.edges + 1
        self.nodes = len(self.V)

    def copy(self):
        G = Graph()
        G.allocate(self.get_size()[0])
        G.E = self.E.copy()
        G.V = self.V.copy()
        G.nodes = self.nodes
        G.edges = self.edges
        return G


class RunExperiments:
    def __init__(self):
        self.MST = Graph()
        self.totalweight = 0
        self.added_edges = []

    def parse_edges(self, filename):
        '''
        parse_edges to form a Graph network, G, that will be processed by
        computeMST()
        :param filename: .gr file with number of vertices and edges in first
        line, and lines of end points and weight
        :return: G graph structure ...
        '''
        # Initialize graph object
        G = Graph()
        # Open file and start reading lines
        if os.path.exists(filename) == False:
            raise FileExistsError("The file {} does not exist!".format(filename))
        with open(filename, 'r') as filereader:
            firstline = filereader.readline().split()
            num_nodes = int(firstline[0])
            num_edges = int(firstline[1])
            G.allocate(num_nodes)
            line = filereader.readline().split()
            while len(line) == 3:
                node1 = int(line[0])
                node2 = int(line[1])
                weight = int(line[2])
                G.insert_edge(node1, node2, weight)
                line = filereader.readline().split()
            filereader.close()
        # print(G.get_size())
        # print(G.E)
        return G


    def computeMST(self, G):
        '''
        computeMST using Prim's Algorithm
        :param G:  Graph Network formed by parse_edges()    1
        :return: Total weight of MST
        '''
        # Initialize cheapest edge value for each node in
        a = np.empty((G.nodes))
        a.fill(np.inf)
        # Initialize set of explored nodes
        S = np.empty((G.get_size()[0]), dtype=bool)
        S.fill(False)  # initialize as not explored
        in_Q = np.empty((G.get_size()[0]), dtype=bool)
        in_Q.fill(False)
        # Initialize priority queue with nodes (Priority is edge weight)
        Q = PriorityQueue()
        # Initialize with first node
        start = 0
        Q.put((0, (start, start), G.V[start]))
        # initialize outputs
        self.MST.allocate(G.get_size()[0])
        W = 0
        tree = []
        costs = []
        fifo = 1
        # print("Starting at node {}".format(start))
        while not Q.empty():
            u_cost, link, u_node = Q.get()
            u = u_node.get_index()
            if S[u]:  # already found the best edge, so throw away
                continue
            tree.append(u)
            costs.append(u_cost)
            if u!=start:
                self.MST.insert_edge(link[0], link[1], u_cost)
            W = W + u_cost
            # print("At node {} with cost {}".format(u, u_cost))
            S[u] = True
            for v in u_node.neighbors:
                cost = G.E[u,v]
                if not S[v] and cost < a[v]:
                    # queue_place = a[v]  # to find
                    a[v] = cost  # update cost
                    # if in_Q[v]:
                    #      for k1,i,k3 in Q.queue:
                    #          if v == i:
                    #             Q.queue.remove(i)
                    #             Q.put((a[v], v, G.V[v]))
                Q.put((a[v], (u, v),   G.V[v]))
                fifo = fifo+1
        self.totalweight = W
        return W

    #
    def computeMST2(self, G):
        '''
        computeMST using Prim's Algorithm
        :param G:  Graph Network formed by parse_edges()    1
        :return: Total weight of MST
        '''

        W = 0

        # Initialize cheapest edge value for each node in

        # Initialize set of explored nodes
        S = np.empty((G.get_size()[0]), dtype=bool)
        S.fill(False)  # initialize as not explored
        in_Q = np.empty((G.get_size()[0]), dtype=bool)
        in_Q.fill(False)

        # Initialize priority queue with nodes (Priority is edge weight)
        # Q = [float("inf")] * G.get_size()[0]
        start = 0
        a = []
        # for i in range(0, G.get_size()[0]):
        #     Q.append((float("inf"), i))
        for i in range(0, G.get_size()[0]):
            a.append([float("inf"), i])
        a[0][0] = 0
        Q = a.copy()
        #Initialize with first node
        start = 0
        tree = []
        costs = []
        print("Starting at node {}".format(start))
        while len(tree) < G.get_size()[0]:
            heapq.heapify(Q)
            cost, u = heapq.heappop(Q)
            tree.append(u)
            costs.append(cost)
            W = W + cost
            print("At node {} with cost {}".format(u, cost))
            S[u] = True
            a[u][0] = float("inf")  # needed?
            for v in G.V[u].neighbors:
                cost = G.E[u,v]
                if not S[v] and cost < a[v][0]:
                    a[v][0] = cost  # update cost
            Q = a.copy()

        self.MST = tree
        return W


    def removecycle_recursive(self, G, u, z, visited, last_edges):
        visited[u] = True
        for v in G.V[u].neighbors:
            if visited[v] == False:
                edges_copy = last_edges.copy()
                edges_copy.append((u,v))
                iscycle, edges = self.removecycle_recursive(G, v, z, visited, edges_copy)
                if iscycle:
                    return True, edges
            elif v == z:
                edges = last_edges.copy()
                edges.append((u,v))
                return True, edges
        return False, last_edges


    def removecycle(self, G, start, end):
        remove = 0
        visited = [False] * G.get_size()[0]
        visited[start] = True
        visited[end] = True
        iscycle, edges = self.removecycle_recursive(G, start, end, visited, [])
        if iscycle:
            #TODO: check cycle for smaller cycle within
            edges.append((start, end))
            costs = []
            max_cost = 0
            max_index = 0
            cnt = 0
            for edge in edges:
                cost = G.E[edge[0],edge[1]]
                if cost > max_cost:
                    max_cost = cost
                    max_index = cnt
                cnt = cnt+1
            remove = max_cost
            removed = edges[max_index]
        return remove, removed

    def recomputeMST(self, u, v, weight, G):
        # Write this function to recompute total weight of MST with the newly added edge
        slow = False
        if slow:
            G.insert_edge(u, v, weight)
            W = self.computeMST(G)
            return W

        if self.MST.E[u,v] != 0:
            if self.MST.E[u,v] <= weight:
                print("Added edge already exists at lower cost in Minimum Spanning Tree!")
                W = self.totalweight
                return W
            else:
                old_weight = self.MST.E[u, v]
                self.MST.E[u, v] = weight
                self.MST.E[v, u] = weight
                W = self.totalweight - old_weight + weight
                self.totalweight = W
                return W

        self.MST.insert_edge(u, v, weight)
        G_MST = self.MST.copy()
        # if G_MST.E[u,v] != 0:
        #     # G_MST.E[u,v] = weight
        #     G.insert_edge(u,v, weight)
        #     W = self.computeMST(G)
        #     # W = self.totalweight
        #     return W
        # if G_MST.E[u,v] != 0 and weight >= G_MST.E[u,v]:
        #     print("Added edge already exists at lower cost in Minimum Spanning Tree!")
        #     return self.totalweight
        # elif G_MST.E[u,v] != 0 and weight < G_MST.E[u,v]:
        #     # old_weight = G_MSTo.E[u,v]
        #     G_MST.E[u,v] = weight
        #     # return self.totalweight - old_weight + weight
        #     W = self.computeMST(G)
        #     return W
        # else:

        # G_MST.insert_edge(u, v, weight)
        if u==3:
            print("stop")
        edge_weight, edge = self.removecycle(G_MST, u, v)
        # update MST #TODO: make it an internal graph call
        if self.MST.E[edge[0], edge[1]] != 0:
            self.MST.remove_edge(edge[0], edge[1])
        #TODO: update graph!
        W = self.totalweight + weight - edge_weight
        self.totalweight = W
        return W


    def main(self):

        num_args = len(sys.argv)

        if num_args < 4:
            print("error: not enough input arguments")
            exit(1)

        graph_file = sys.argv[1]
        change_file = sys.argv[2]
        output_file = sys.argv[3]

        # Construct graph
        G = self.parse_edges(graph_file)

        start_MST = time.time()
        # call MST function to return total weight of MST
        MSTweight = self.computeMST(G)
        end_MST = time.time()

        total_time = (end_MST - start_MST) * 1000

        # Write initial MST weight and time to file
        output = open(output_file, 'w')
        output.write(str(MSTweight) + " " + str(total_time) + "\n")


        # Changes file
        with open(change_file, 'r') as changes:
            num_changes = changes.readline()

            for line in changes:
                # parse edge and weight
                edge_data = list(map(lambda x: int(x), line.split()))
                assert(len(edge_data) == 3)

                u, v, weight = edge_data[0], edge_data[1], edge_data[2]

                # call recomputeMST function
                start_recompute = time.time()
                new_weight = self.recomputeMST(u, v, weight, G)
                # to convert to milliseconds
                end_recompute = time.time()
                total_recompute = (end_recompute - start_recompute) * 1000


                # write new weight and time to output file
                output.write(str(new_weight) + " " + str(total_recompute) + "\n")


if __name__ == '__main__':
    # run the experiments
    runexp = RunExperiments()
    runexp.main()
