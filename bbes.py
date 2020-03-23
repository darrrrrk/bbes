from __future__ import print_function

import numpy as np
import Queue
import math
import collections
import warnings
import pickle
import time
import sys
import math
from UnionFind import UnionFind # from David Eppstein's PADS library

def node_name(v):
    return chr(97 + v)

def node_name_list(S):
    ret = ""
    for v, val in enumerate(S):
        if val:
            ret += node_name(v)
    return ret

def bin2mask(d, bin):
    mask = np.zeros(d, dtype=bool)
    for i in range(d):
        if bin & (1 << i):
            mask[i] = True
    return mask

def mask2bin(mask):
    #    self.powersof2 = np.ones(self.d, dtype=int)
    #    for i in range(1, self.d):
    #        self.powersof2[i] = self.powersof2[i-1] << 1
    # ...
    # int(self.powersof2.dot(mask))
    bin = 0
    for i, val in enumerate(mask):
        if val:
            bin += (1 << i)
    return bin

class EquivalenceClass:
    """A Markov equivalence class (set of DAGs) as point in the search space"""
    def __init__(self, pdag, essential_mB=None, children=None):
        if essential_mB is not None:
            # with precomputed search space
            self.example_mB = pdag # so should not contain undir edges here
            self.num_edges = np.count_nonzero(self.example_mB)
            (self.skeleton, self.vstructs) = skeleton_vstructs_normalizer(self.example_mB)
            self.essential_mB = essential_mB # AND of mB's
            self.children = children
            self.children_identifying_constraint = None # computed later
        else:
            # without precomputed search space: less data
            self.pdag = pdag
            self.is_pdag_completed = False # otherwise, it is minimal
            self.num_edges = np.count_nonzero(np.logical_or(pdag, pdag.T)) / 2
    def get_cpdag(self):
        if hasattr(self, 'essential_mB'):
            # precomputed case
            d = self.example_mB.shape[0]
            return np.logical_and(np.logical_and(self.skeleton, np.logical_not
                                                 (np.eye(d, dtype=bool))),
                                  np.logical_not(self.essential_mB.T))
        # non-precomputed case
        if not self.is_pdag_completed:
            self.pdag = complete_pdag(self.pdag)
            self.is_pdag_completed = True
        return self.pdag
    def __str__(self):
        if hasattr(self, 'essential_mB'):
            d = self.example_mB.shape[0]
            ret = ""
            for v in range(d):
                for w in range(d):
                    if v == w:
                        ret += " O  "
                    elif not self.skeleton[v,w]:
                        ret += " .  "
                    elif self.vstructs[v,w]:
                        ret += "<-- "
                    elif self.vstructs[w,v]:
                        ret += "--> "
                    elif self.essential_mB[v,w] and not self.essential_mB[w,v]:
                        ret += "{-- " # direction inferred indirectly
                    elif self.essential_mB[w,v] and not self.essential_mB[v,w]:
                        ret += "--} " # direction inferred indirectly
                    else:
                        ret += "--- "
                if v < d - 1: # no \n at end, as a print will usually add it
                    ret += "\n"
            #ret += str(self.children) + "\n"
            return ret
        else:
            d = self.pdag.shape[0]
            ret = ""
            for v in range(d):
                for w in range(d):
                    if v == w:
                        ret += " O  "
                    elif not self.pdag[v,w] and not self.pdag[w,v]:
                        ret += " .  "
                    elif self.pdag[v,w] and not self.pdag[w,v]:
                        ret += "<-- "
                    elif self.pdag[w,v] and not self.pdag[v,w]:
                        ret += "--> "
                    else:
                        ret += "--- "
                if v < d - 1:
                    # no terminating \n, as a print will usually add it
                    ret += "\n"
            #ret += str(self.children) + "\n"
            if not self.is_pdag_completed:
                ret += "(may not be complete)\n"
            return ret

def saturated_class(d):
    pdag = np.logical_not(np.eye(d, dtype=bool))
    ret = EquivalenceClass(pdag)
    ret.is_pdag_completed = True
    return ret

class Constraint:
    """A conditional independence constraint"""
    def __init__(self, v, w, S, imposed_by=None):
        # NOTE: order of v and w may matter when using constraints to represent
        # Delete operators (because only one of the two might be defined; if
        # both are defined and valid, they represent the same operator).
        self.v = v
        self.w = w
        self.S = S # mask (i.e. np.array of bools)
        self.imposed_by = imposed_by
    def has_same_adjacency(self, other):
        if self.v == other.v and self.w == other.w:
            return True
        if self.v == other.w and self.w == other.v:
            return True
        return False
    def is_same_constraint(self, other):
        if not self.has_same_adjacency(other):
            return False
        return np.all(self.S == other.S)
    def __str__(self):
        ret = "{0} _||_ {1}".format(node_name(self.v), node_name(self.w))
        if np.any(self.S):
            ret += " |"
            for i in range(self.S.size):
                if self.S[i]:
                    ret += " {0}".format(node_name(i))
        return ret

def transitive_reflexive_closure(mB):
    # (from aelsem)
    d = mB.shape[0]
    trans_refl_closure = np.logical_or(np.eye(d, dtype=bool), mB)
    prev_num = np.count_nonzero(trans_refl_closure)
    # O(log n) calls to numpy matrix multiplication probably faster than
    # O(n^3) loop in Python
    while True:
        #print(trans_refl_closure)
        trans_refl_closure = np.linalg.matrix_power(trans_refl_closure, 2)
        num = np.count_nonzero(trans_refl_closure)
        if num == prev_num:
            break
        prev_num = num
    return trans_refl_closure

def has_cycles(mB):
    # (from aelsem)
    # test for cycles of length 2 or more
    d = mB.shape[0]
    trans_edges = np.logical_xor(transitive_reflexive_closure(mB),
                                 np.eye(d, dtype=bool))
    return np.any(np.logical_and(trans_edges, trans_edges.T))

def hash_model((skel, vstr)):
    return skel.tostring() + vstr.tostring()

def skeleton_vstructs_normalizer(mB):
    d = mB.shape[0]
    skel = np.logical_or(np.eye(d, dtype=bool), np.logical_or(mB, mB.T))
    # mark arrows v<--w for which also v<--x...w (and x != w: uses that skel
    # is True on diagonal)
    vstr = np.logical_and(mB, mB.dot(np.logical_not(skel)))
    return (skel, vstr)

def complete_pdag(pdag):
    # TODO: other algorithms exist for this task that are in principle more
    # efficient; see e.g. Chickering2002
    d = pdag.shape[0]
    # set diagonal to False (skeleton_vstructs_normalizer sets it to True
    # on skel)
    pdag = np.logical_and(pdag, np.logical_not(np.eye(d, dtype=bool)))
    skel = np.logical_or(pdag, pdag.T)
    mB = np.logical_and(pdag, np.logical_not(pdag.T))
    mU = np.logical_and(pdag, pdag.T)
    undir_count = np.sum(mU)
    while undir_count > 0:
        # Keep applying Meek's orientation rules until nothing changes
        # Rule 1: v---w with v...x-->w becomes v<--w
        mB_add = np.logical_and(mU, np.dot(np.logical_not(skel), mB.T))
        np.logical_or(mB, mB_add, out=mB)
        np.logical_and(mU, np.logical_not(mB_add), out=mU)
        np.logical_and(mU, np.logical_not(mB_add).T, out=mU)
        # Rule 2: v---w with v<--x<--w becomes v<--w
        mB_add = np.logical_and(mU, np.dot(mB, mB))
        np.logical_or(mB, mB_add, out=mB)
        np.logical_and(mU, np.logical_not(mB_add), out=mU)
        np.logical_and(mU, np.logical_not(mB_add).T, out=mU)
        for x in range(d):
            # Rule 3: v---w with v<--x---w and v<--y---w but x...y becomes v<--w
            valid_y = np.logical_not(skel[x,:])
            valid_y[x] = False
            mB_add = np.logical_and(np.logical_and(mU, np.outer(mB[:,x],
                                                                mU[x,:])),
                                    np.dot(np.logical_and(mB, valid_y),
                                           mU))
            np.logical_or(mB, mB_add, out=mB)
            np.logical_and(mU, np.logical_not(mB_add), out=mU)
            np.logical_and(mU, np.logical_not(mB_add).T, out=mU)
            # Rule 4: v---w with v...x---w and v<--y*-*w and y<--x becomes v<--w
            mB_add = np.logical_and(np.logical_and(mU, np.outer(np.logical_not(skel[:,x]), mU[x,:])),
                                    np.dot(np.logical_and(mB, mB[:,x]),
                                           mU))
            np.logical_or(mB, mB_add, out=mB)
            np.logical_and(mU, np.logical_not(mB_add), out=mU)
            np.logical_and(mU, np.logical_not(mB_add).T, out=mU)
        new_undir_count = np.sum(mU)
        if new_undir_count == undir_count:
            break
        undir_count = new_undir_count
    pdag = np.logical_or(mB, mU)
    return pdag

def complete_pdag_fixed_orientations(orig_cpdag, req_adjs):
    # Using v-structures and Meek's orientation rules, find what orientations
    # are fixed in a set of classes defined by an original cpdag with some
    # edges not required (i.e. might be deleted).
    # * Return value fixed_orient will be a subset of the cpdag's mB part.
    # * A directed edge that is not required can also be marked as "fixed",
    #   which means that it has that orientation in all classes where the
    #   two nodes are adjacent.
    d = orig_cpdag.shape[0]
    orig_adj = np.logical_or(orig_cpdag, orig_cpdag.T)
    orig_mU = np.logical_and(orig_cpdag, orig_cpdag.T)
    sure_nonadj = np.logical_not(np.logical_or(orig_adj,
                                               np.eye(d, dtype=bool)))
    orig_mB = np.logical_and(orig_cpdag, np.logical_not(orig_cpdag.T))
    req_mB = np.logical_and(np.logical_not(orig_cpdag.T), req_adjs)
    # If an arrow v-->w participates in a guaranteed v-structure (x-->w
    # required, v and x definitely not adjacent), then its orientation is fixed
    # in all cpdags where v and w are adjacent. (If v-->w is required, then
    # x-->w will be marked as fixed by the same rule; if it isn't, it's still
    # correct to mark v-->w as fixed. This is the converse of Chickering2002's
    # Lemma 28; see notes on printout for proof.)
    fixed_orient = np.logical_and(orig_mB, #w<--v
                                  (req_mB) # w<--x req
                                  .dot(sure_nonadj)) # x...v sure

    #orig_skel = np.logical_or(orig_cpdag, orig_pdag.T)
    #orig_mB = np.logical_and(orig_cpdag, np.logical_not(orig_cpdag.T))
    #mU = np.logical_and(pdag, pdag.T)
    fixed_count = np.sum(fixed_orient)
    while True:
        # Keep applying Meek's orientation rules until nothing changes
        # Rule 1: v<--w with v...x [sure] and x-->w [req,fixed] becomes fixed
        fix_add = np.logical_and(orig_mB,
                                 np.dot(sure_nonadj,
                                        np.logical_and(fixed_orient.T,
                                                       req_adjs)))
        np.logical_or(fixed_orient, fix_add, out=fixed_orient)
        # Rule 2: v<--w with v<--x [req,fixed] and x<--w [fixed] becomes fixed
        # (x<--w not required: if absent, v<--w is still fixed because it is
        # in a v-structure)
        fix_add = np.logical_and(orig_mB,
                                 np.dot(np.logical_and(fixed_orient, req_adjs),
                                        fixed_orient))
        np.logical_or(fixed_orient, fix_add, out=fixed_orient)
        for x in range(d):
            # Rule 3: v<--w with v<--x [req,fixed], x---w and
            # v<--y [req, fixed], y---w but x...y [sure] becomes fixed
            valid_y = sure_nonadj[x,:]
            fix_add = np.logical_and(np.logical_and(orig_mB,
                                                    np.outer(np.logical_and(fixed_orient[:,x], req_adjs[:,x]),
                                                             orig_mU[x,:])),
                                     np.dot(np.logical_and(np.logical_and(fixed_orient, req_adjs), valid_y),
                                            orig_mU))
            np.logical_or(fixed_orient, fix_add, out=fixed_orient)
            # Rule 4: v---w with v...x---w and v<--y*-*w and y<--x becomes v<--w
            # TODO
            #fix_add = np.logical_and(np.logical_and(mU, np.outer(np.logical_not(skel[:,x]), mU[x,:])),
            #                         np.dot(np.logical_and(mB, mB[:,x]),
            #                                mU))
            #np.logical_or(fixed_orient, fix_add, out=fixed_orient)
        new_fixed_count = np.sum(fixed_orient)
        if new_fixed_count == fixed_count:
            break
        fixed_count = new_fixed_count
    return fixed_orient

def complete_pdag_simpleTEST():
    rule3 = np.ones((4,4), dtype=bool)
    rule3[2,3] = rule3[3,2] = False
    rule3[2,0] = rule3[3,0] = False
    print("Rule 3:")
    print(rule3)
    print(complete_pdag(rule3))
    rule4 = np.ones((4,4), dtype=bool)
    rule4[2,3] = rule4[3,2] = False
    rule4[1,2] = rule4[3,1] = False
    print("Rule 4:")
    print(rule4)
    print(complete_pdag(rule4))
    rule4[0,1] = False
    print("Rule 4b:")
    print(rule4)
    print(complete_pdag(rule4))
    rule4[0,1] = True
    rule4[1,0] = False
    print("Rule 4c:")
    print(rule4)
    print(complete_pdag(rule4))

def complete_pdag_bigTEST(eq_classes):
    # Tested for d=4 and d=5: correct
    print("Testing complete_pdag()")
    d = eq_classes[0].skeleton.shape[0]
    for eq_class in eq_classes:
        #pattern = np.logical_or(eq_class.vstructs, eq_class.skeleton)
        pattern = np.logical_and(eq_class.skeleton, np.logical_not(eq_class.vstructs.T))
        cpdag = complete_pdag(pattern)
        #cpdag_expected = np.logical_and(np.logical_or(eq_class.essential_mB,
        #                                              eq_class.skeleton),
        #                                np.logical_not(np.eye(d, dtype=bool)))
        cpdag_expected = np.logical_and(np.logical_and(np.logical_not(eq_class.essential_mB).T,
                                                       eq_class.skeleton),
                                        np.logical_not(np.eye(d, dtype=bool)))
        if np.any(cpdag != cpdag_expected):
            print("Error")
            print(eq_class)
            print(cpdag)
            print(cpdag_expected)
            return

def generate_valid_delete_operators(cpdag):
    # TODO: this can probably be made more efficient using the original class's
    # list of delete operators
    # TODO: only generate operators up to some conditioning set size
    d = cpdag.shape[0]
    ret = []
    for v in range(d-1):
        for w in range(v+1, d):
            if cpdag[v,w] == False and cpdag[w,v] == False:
                continue
            # opt_vw = NA_YX; req_vw = Pa_Y \ X
            if cpdag[w,v] == True:
                # v --> w or v --- w
                # can take x=v, y=w
                opt_vw = np.logical_and(np.logical_or(cpdag[v,:],
                                                      cpdag[:,v]), # na *-* x
                                        np.logical_and(cpdag[w,:],
                                                       cpdag[:,w])) # na --- y
                req_vw = np.logical_and(cpdag[w,:], np.logical_not(cpdag[:,w]))
                req_vw[v] = False
                opt_vw_list = np.nonzero(opt_vw)[0]
                n = len(opt_vw_list)
                for opt_mask in range(1 << n):
                    S = req_vw.copy()
                    for i in range(n):
                        if (opt_mask & (1 << i)):
                            S[opt_vw_list[i]] = True
                    new_constraint = Constraint(v, w, S)
                    if is_delete_operator_valid(cpdag, new_constraint):
                        ret.append(new_constraint)
            if cpdag[v,w] == True:
                # w --> v or w --- v
                # can take x=w, y=v
                opt_wv = np.logical_and(np.logical_or(cpdag[w,:],
                                                      cpdag[:,w]), # na *-* x
                                        np.logical_and(cpdag[v,:],
                                                       cpdag[:,v])) # na --- y
                req_wv = np.logical_and(cpdag[v,:], np.logical_not(cpdag[:,v]))
                req_wv[w] = False
                opt_wv_list = np.nonzero(opt_wv)[0]
                n = len(opt_wv_list)
                for opt_mask in range(1 << n):
                    S = req_wv.copy()
                    for i in range(n):
                        if (opt_mask & (1 << i)):
                            S[opt_wv_list[i]] = True
                    if cpdag[w,v] == True:
                        # TODO: testing if this can be simplified:
                        if not (np.all(req_vw == req_wv)
                                and np.all(opt_vw == opt_wv)):
                            print("WARNING: Different set of existing Delete operators for {0}---{1} vs {1}---{0}"
                                  .format(node_name(v), node_name(w)))
                            print("Required nodes:")
                            print(req_vw)
                            print(req_wv)
                            print("Optional nodes:")
                            print(opt_vw)
                            print(opt_wv)
                        # v --- w, so pay attention to avoid duplicate operators
                        duplicate = True
                        if np.any(np.logical_and(req_vw, np.logical_not(S))):
                            duplicate = False
                        if np.any(np.logical_and(S, np.logical_not(np.logical_or(opt_vw, req_vw)))):
                            duplicate = False
                        if duplicate:
                            continue
                    new_constraint = Constraint(w, v, S)
                    if is_delete_operator_valid(cpdag, new_constraint):
                        ret.append(new_constraint)
    return ret

def is_delete_operator_valid(cpdag, constraint):
    d = cpdag.shape[0]
    x = constraint.v
    y = constraint.w
    # Chickering2002: valid iff NA_YX \ H is a clique
    # my S = Chickering's NA(Y,X) \ H cup Pa_Y (where NA(Y,X) and Pa_Y disjoint)
    # So: NA_YX \ H = S \ Pa_Y
    clique = np.logical_and(constraint.S,
                            np.logical_not(np.logical_and
                                           (cpdag[y,:],
                                            np.logical_not(cpdag[:,y]))))
    # add diagonal
    cpdag_diag = np.logical_or(cpdag, np.eye(d, dtype=bool))
    return np.all(np.logical_or(cpdag_diag, cpdag_diag.T)[clique,:][:,clique])

def apply_delete_operator(cpdag, constraint):
    # Returns the *minimal* PDAG (i.e. a pattern) representing the equivalence
    # class obtained by taking a CPDAG, and applying the Delete operator
    # corresponding to the given constraint.
    pdag = cpdag.copy()
    # Assumes that constraint specifies a valid and defined Delete operator
    # in the sense of Chickering2002, for x=v and y=w (i.e. order matters).
    x = constraint.v
    y = constraint.w
    # my S = Chickering's NA(Y,X) \ H cup Pa_Y (where NA(Y,X) and Pa_Y disjoint)
    # So: H = NA(Y,X) \ S
    NA_YX = np.logical_and(np.logical_or(pdag[x,:], pdag[:,x]), # na *-* x
                           np.logical_and(pdag[y,:], pdag[:,y])) # na --- y
    H = np.logical_and(NA_YX, np.logical_not(constraint.S))
    # (if pdag has True on diagonal [shouldn't happen], then that won't hurt)
    pdag[x,y] = pdag[y,x] = False
    pdag[x,H] = pdag[y,H] = False
    # This PDAG may have directed arrows that could be oriented the other way
    # around in some equivalent DAG (a PDAG just needs to have the right
    # skeleton and v-structures).
    # Generalization of skeleton_vstructs_normalizer():
    mB = np.logical_and(pdag, np.logical_not(pdag.T))
    nonadj = np.logical_not(np.logical_or(np.logical_or(pdag, pdag.T),
                                          np.eye(pdag.shape[0], dtype=bool)))
    vstr = np.logical_and(mB, mB.dot(nonadj))
    pdag = np.logical_and(np.logical_or(pdag, pdag.T),
                          np.logical_not(vstr.T))
    return pdag

def generate_delete_operators_TEST():
    # check output on cpdag that imposes only a _||_ b | c
    d = 4
    saturated_cpdag = saturated_class(d).get_cpdag()
    constraint = Constraint(0, 1, bin2mask(d, 4))
    pdag = apply_delete_operator(saturated_cpdag, constraint)
    cpdag = complete_pdag(pdag)
    res = generate_valid_delete_operators(cpdag)
    for del_op in res:
        print(del_op)

def partition_graphs(graphs, normalizer, verbose=False):
    classes = {}
    ret = []
    for graph in graphs:
        which_class = normalizer(graph)
        #print(graph)
        #print("normalizes to")
        #print(which_class)
        #print(".")
        if hash_model(which_class) in classes:
            classes[hash_model(which_class)].append(graph)
        else:
            classes[hash_model(which_class)] = [graph]
            ret.append(which_class)
    if verbose:
        print("Identified", len(ret), "distinct classes using",
              normalizer.__name__, file=sys.stderr)
    for i, which_class in enumerate(ret):
        ret[i] = classes[hash_model(which_class)]
    return ret

def generate_all_DAGs(d):
    graphs = []
    n = d * (d-1) / 2
    N = 3 ** n
    print("Generating DAGs:", file=sys.stderr)
    for mask in range(N):
        if (mask & 255) == 0:
            print("\r{0:0.2f}% - generated {1} DAGs"
                  .format(100.0 * mask / N, len(graphs)),
                  end='', file=sys.stderr)
        mB = np.zeros((d,d), dtype=bool)
        for i in range(1, d):
            for j in range(0, i):
                mask, curmask = divmod(mask, 3)
                if curmask == 1:
                    mB[i,j] = True
                elif curmask == 2:
                    mB[j,i] = True
        if not has_cycles(mB):
            graphs.append(mB)
    print("\r100.00%; generated", len(graphs), "DAGs", file=sys.stderr)
    return graphs

def compute_essential_mB(graphs):
    ret = graphs[0]
    for graph in graphs:
        ret = np.logical_and(ret, graph)
    return ret

def is_d_connected_dfs(mB, pos, w, S, vis):
    # Made modifications to deal with CPDAGs as input
    #print("At ", pos)
    (v, dir) = pos
    if v == w:
        return True
    if (dir == 0 and not S[v]) or (dir == 1 and S[v]):
        # traverse backward (dir=0) along an arrow
        #print(mB)
        #print(v)
        #print(mB[v,:])
        #print(vis[:,0])
        next_vs_mask = np.logical_and(mB[v,:], np.logical_not(vis[:,0]))
        if dir == 1:
            # we can't continue on an undirected path in case dir == 1 and S[v])
            next_vs_mask = np.logical_and(next_vs_mask, np.logical_not(mB[:,v]))
        #print(next_vs_mask.shape)
        tmp = np.logical_or(vis[:,0], next_vs_mask)
        #print(tmp.shape)
        vis[:,0] = tmp
        #print(np.nonzero(next_vs_mask))
        #print(np.nonzero(next_vs_mask)[0])
        for next_v in np.nonzero(next_vs_mask)[0]:
            #print("visiting ", next_v)
            if is_d_connected_dfs(mB, (next_v, 0), w, S, vis):
                return True
    if not S[v]:
        # traverse forward (dir=1) along an arrow
        next_vs_mask = np.logical_and(mB[:,v], np.logical_not(vis[:,1]))
        next_vs_mask = np.logical_and(next_vs_mask, np.logical_not(mB[v,:]))
        vis[:,1] = np.logical_or(vis[:,1], next_vs_mask)
        for next_v in np.nonzero(next_vs_mask)[0]:
            if is_d_connected_dfs(mB, (next_v, 1), w, S, vis):
                return True
    return False

def is_d_separated(mB, v, w, S):
    # mB can also be a cpdag
    if S[v] or S[w]:
        return True
    d = mB.shape[0]
    # vis[v,0]: reachable by path ending in tail
    # vis[v,1]: reachable by path ending in head
    vis = np.zeros((d,2), dtype=bool)
    pos = (v,0)
    vis[pos] = True
    is_d_connected_dfs(mB, pos, w, S, vis)
    if vis[w,0] or vis[w,1]:
        return False
    return True

def is_d_separated_TEST():
    returned = []
    expected = []
    d = 4
    S = np.zeros(d, dtype=bool)
    S1 = np.copy(S)
    S1[1] = True
    S2 = np.copy(S)
    S2[2] = True
    S12 = np.copy(S)
    S12[1] = S12[2] = True
    # 0 <-- 1 --> 2 --> 3
    mB = np.zeros((d,d), dtype=bool)
    mB[0,1] = mB[2,1] = mB[3,2] = True
    returned.append(is_d_separated(mB, 0, 3, S))
    expected.append(False)
    returned.append(is_d_separated(mB, 0, 3, S1))
    expected.append(True)
    returned.append(is_d_separated(mB, 0, 3, S2))
    expected.append(True)
    returned.append(is_d_separated(mB, 0, 3, S12))
    expected.append(True)
    # 0 --> 1 <-- 3 with 1 --> 2
    mB = np.zeros((d,d), dtype=bool)
    mB[1,0] = mB[2,1] = mB[1,3] = True
    returned.append(is_d_separated(mB, 0, 3, S))
    expected.append(True)
    returned.append(is_d_separated(mB, 0, 3, S1))
    expected.append(False)
    returned.append(is_d_separated(mB, 0, 3, S2))
    expected.append(False)
    returned.append(is_d_separated(mB, 0, 3, S12))
    expected.append(False)
    if returned != expected:
        print("There were incorrect results in is_d_separated:")
        print("Return values:")
        print(returned)
        print("Expected:")
        print(expected)
    else:
        print("All tests passed for is_d_separated")

def lemma_34_TEST(d):
    # Test Lemma 34 from Chickering: if two edges in a clique of size three
    # in a CPDAG are undirected, then so is the third.
    solver = BBES(d, 0)
    equivalence_classes = solver.search_space['equivalence_classes']
    for eq_class in equivalence_classes:
        cpdag = eq_class.get_cpdag()
        for v in range(d-1):
            for w in range(v+1, d):
                if cpdag[v,w] and cpdag[w,v]:
                    for x in range(d):
                        num = 1
                        if cpdag[v,x] and cpdag[x,v]:
                            num += 1
                        if cpdag[w,x] and cpdag[x,w]:
                            num += 1
                        if ((cpdag[v,x] or cpdag[x,v])
                            and (cpdag[w,x] or cpdag[x,w])
                            and num == 2):
                            print("ERROR: clique {0}-{1}-{2} violates Chickering's Lemma 34 in the following CPDAG:"
                                  .format(v, w, x))
                            print(eq_class)
    print("Test of Lemma 34 complete")
    # Test completed with no errors on d=4,5,6

def canonical_constraint(mB_parent_class, mB_child_class):
    # Find the constraint that a child class introduces compared to a parent
    # class, such that the difference in likelihood of those classes equals the
    # difference between a class imposing only this constraint and the saturated
    # class.
    # This is not necessarily the "simplest" constraint (but it doesn't need
    # to be; it mostly needs to be easy to compute). E.g. to separate
    # parent a --> b <-- c from child a --- b     c, the constraint is
    # b _||_ c | a.
    d = mB_parent_class.shape[0]
    error_msg = None
    coefficients = collections.Counter()
    for v in range(d):
        parents = mB_parent_class[v,:]
        bin_parents = mask2bin(parents)
        bin_parents_and_v = bin_parents + (1 << v)
        coefficients[bin_parents_and_v] += 1
        coefficients[bin_parents] -= 1
    for v in range(d):
        parents = mB_child_class[v,:]
        bin_parents = mask2bin(parents)
        bin_parents_and_v = bin_parents + (1 << v)
        # as above, with + and - reversed
        coefficients[bin_parents_and_v] -= 1
        coefficients[bin_parents] += 1
    positive_bins = []
    for bin, coefficient in coefficients.items():
        if coefficient <= 0:
            continue
        # there should be two entries with positive coefficient (namely 1):
        # S and S+{v}+{w}
        if coefficient > 1:
            error_msg = "canonical_constraint encountered cluster with coefficient 2"
            break
        positive_bins.append(bin)
    if len(positive_bins) != 2:
        error_msg = "canonical_constraint encountered {0} clusters with coefficient 1 (expected 2)".format(len(positive_bins))
    else:
        positive_bins.sort()
        S = bin2mask(d, positive_bins[0])
        Svw = bin2mask(d, positive_bins[1])
        if np.any(np.logical_and(S, np.logical_not(Svw))):
            error_msg = "canonical_constraint encountered pair of coef-1 clusters that are not subset-related"
        elif np.sum(np.logical_and(Svw, np.logical_not(S))) != 2:
            error_msg = "canonical_constraint encountered pair of coef-1 clusters that differ by other than 2 elements"
        else:
            vw_list = np.nonzero(np.logical_and(Svw, np.logical_not(S)))[0]
            v = vw_list[0]
            w = vw_list[1]
            return Constraint(v, w, S)
    # Something unexpected occured:
    print("canonical_constraint - mB_parent_class:", file=sys.stderr)
    print(mB_parent_class, file=sys.stderr)
    print("canonical_constraint - mB_child_class:", file=sys.stderr)
    print(mB_child_class, file=sys.stderr)
    raise ValueError(error_msg)

def find_constraint_in_list(constraints, constraint):
    # Binary search using that constraints in list are sorted on (v, w, S_bin)
    v = constraint.v
    w = constraint.w
    S_bin = mask2bin(constraint.S)
    lo = 0
    hi = len(constraints)
    while lo < hi:
        mid = lo + (hi - lo) / 2
        if (constraints[mid].v > v
            or (constraints[mid].v == v and constraints[mid].w > w)
            or (constraints[mid].v == v and constraints[mid].w == w
                and mask2bin(constraints[mid].S) >= S_bin)):
            # constraints[mid] >= constraint
            hi = mid
        else:
            lo = mid + 1
    if lo >= len(constraints):
        raise ValueError("constraint not found")
    return lo

def search_space_precompute(filename, d):
    all_graphs = generate_all_DAGs(d)
    graphs_by_class = partition_graphs(all_graphs, skeleton_vstructs_normalizer,
                                       verbose=True)
    # sort by number of edges
    print("Sorting equivalence classes by number of edges", file=sys.stderr)
    graphs_by_class = sorted(graphs_by_class,
                             key=lambda graphs: np.count_nonzero(graphs[0]))
    print("Computing EquivalenceClass list", file=sys.stderr)
    equivalence_classes = [EquivalenceClass(graphs[0],
                                            compute_essential_mB(graphs),
                                            [])
                           for graphs in graphs_by_class]
    first_class_with_num_params = np.zeros(equivalence_classes[-1].num_edges + 2, dtype=int)
    cur_num_params = -1
    for i, eq_class in enumerate(equivalence_classes):
        num_params_here = eq_class.num_edges
        if num_params_here > cur_num_params:
            cur_num_params = num_params_here
            first_class_with_num_params[cur_num_params] = i
    first_class_with_num_params[-1] = len(equivalence_classes)
    #print(first_class_with_num_params)

    # compute children of each equivalence class
    print("Computing child classes of each equivalence class:", file=sys.stderr)
    index_of_skel_vstr = {hash_model(skeleton_vstructs_normalizer(eq_class.example_mB)):index for index, eq_class in enumerate(equivalence_classes)}
    for parent_index, parent_graphs in enumerate(graphs_by_class):
        children = set()
        for parent_graph in parent_graphs:
            for v in range(d):
                for w in range(d):
                    if parent_graph[v,w]:
                        child_graph = parent_graph.copy()
                        child_graph[v,w] = False
                        child_index = index_of_skel_vstr[hash_model(skeleton_vstructs_normalizer(child_graph))]
                        children.add(child_index)
        equivalence_classes[parent_index].children = sorted(children)
        print("\r{0}: {1:0.2f}%"
              .format(parent_index + 1,
                      100.0 * (parent_index + 1) / len(equivalence_classes)),
              end='', file=sys.stderr)
    print(file=sys.stderr)

    num_constraints = d * (d-1) / 2 * (1 << (d-2))
    num_computed = 0
    constraints = []
    print("Computing constraints:", file=sys.stderr)
    # For n=6, this takes about 5 hours
    # file sizes (with example_mB not set to bool in first version):
    # n=3: 5 KB -> 4 KB
    # n=4: 91 KB -> 70 KB
    # n=5: 5.6 MB -> 4.1 MB
    # n=6: 1.01 GB -> .77 GB
    for v in range(d-1):
        for w in range(v+1, d):
            for S_bin in range(1 << d):
                if (S_bin & (1 << v)) or (S_bin & (1 << w)):
                    continue
                num_computed += 1
                S = np.zeros(d, dtype=bool)
                for i in range(d):
                    if S_bin & (1 << i):
                        S[i] = True
                imposed_by = np.zeros(len(equivalence_classes), dtype=bool)
                for i, eq_class in enumerate(equivalence_classes):
                    if is_d_separated(eq_class.example_mB, v, w, S):
                        imposed_by[i] = True
                constraint = Constraint(v, w, S, imposed_by)
                constraints.append(constraint)
                #print("Constraint", constraint, "is imposed by:")
                #print(imposed_by)
                print("\r{0}: {1:0.2f}%"
                      .format(num_computed,
                              100.0 * num_computed / num_constraints),
                      sep='', end='', file=sys.stderr)
    print(file=sys.stderr)

    print("Computing identifying constraint for each parent-child pair:",
          file=sys.stderr)
    #unique_constraints_TEST(equivalence_classes, constraints,
    #                        verbose=False, write=True)
    for parent_i, parent_class in enumerate(equivalence_classes):
        children_identifying_constraint = []
        for child_list_i, child_i in enumerate(parent_class.children):
            child_class = equivalence_classes[child_i]
            likelihood_ratio_constraint = canonical_constraint(parent_class.example_mB, child_class.example_mB)
            constraint_i = find_constraint_in_list(constraints, likelihood_ratio_constraint)
            children_identifying_constraint.append(constraint_i)
        parent_class.children_identifying_constraint = children_identifying_constraint
        print("\r{0}: {1:0.2f}%"
              .format(parent_i + 1,
                      100.0 * (parent_i + 1) / len(equivalence_classes)),
              end='', file=sys.stderr)
    print(file=sys.stderr)

    if False:
        # Print a list of all classes, with their children + id-constraints
        for i, eq_class in enumerate(equivalence_classes):
            print(i)
            print(eq_class)
            for j, child_i in enumerate(eq_class.children):
                print("Child {0} with identifying constraint {1}"
                      .format(child_i,
                              constraints[eq_class.children_identifying_constraint[j]]))

    search_space = dict(equivalence_classes=equivalence_classes,
                        constraints=constraints,
                        first_class_with_num_params=first_class_with_num_params)
    print("Saving results to file {0}".format(filename),
          file=sys.stderr)
    with open(filename, "wb") as pickle_file:
        pickle.dump(search_space, pickle_file, pickle.HIGHEST_PROTOCOL)
    return search_space

def search_space_prepare(d):
    filename = ("DAG_classes_{0}.pickle".format(d))
    try:
        pickle_file = open(filename, "rb")
        search_space = pickle.load(pickle_file)
    except IOError:
        print("Precomputing search space data; will store it in {0}"
              .format(filename), file=sys.stderr)
        search_space = search_space_precompute(filename, d)
    else:
        pickle_file.close()

    return search_space

def get_descendant_classes(eq_classes, ancestor_i):
    # Boolean vector indicating the descendants of an equivalence class,
    # including that class itself
    desc = np.zeros(len(eq_classes), dtype=bool)
    desc[ancestor_i] = True
    for i in range(ancestor_i, -1, -1):
        if not desc[i]:
            continue
        for child_i in eq_classes[i].children:
            desc[child_i] = True
    return desc

def unique_constraints_TEST(eq_classes, constraints, verbose=False, write=False):
    # For a given parent and child class, a "unique constraint" is one that is
    # imposed by the child class, but not by the parent or any of the parent's
    # other children. We call such a constraint "bad" if it is imposed by
    # some descendant of the parent class other than the child class or one of
    # its descendants. This function tests whether the precomputed "identifying
    # constraint" is unique and not bad; if no id-constraint was precomputed
    # yet, just check if a non-bad unique constraint exists (and if write=True,
    # also write such constraints to eq_classes).
    # Fully tested for d=4,5 (succes); for d=6, only tested up to 33.44%
    print("unique_constraints_TEST:", file=sys.stderr)
    for parent_i, parent_class in enumerate(eq_classes):
        children_identifying_constraint = []
        constraints_parent = np.zeros(len(constraints), dtype=bool)
        for j in range(len(constraints)):
            constraints_parent[j] = constraints[j].imposed_by[parent_i]
        descendants_of_parent = get_descendant_classes(eq_classes, parent_i)
        constraints_of_children = []
        #constraints_union = np.zeros(len(constraints), dtype=bool)
        #constraints_multiple = np.zeros(len(constraints), dtype=bool)
        constraints_union = constraints_parent
        constraints_multiple = constraints_parent
        for child_i in parent_class.children:
            #print(parent_i, child_i)
            constraints_child = np.zeros(len(constraints), dtype=bool)
            for j in range(len(constraints)):
                constraints_child[j] = constraints[j].imposed_by[child_i]
            constraints_of_children.append(constraints_child)
            if np.any(np.logical_and(constraints_parent,
                                     np.logical_not(constraints_child))):
                print("Error: parent {0} imposes a constraint not imposed by child {1}".format(parent_i, child_i))
            constraints_multiple = (np.logical_or
                                    (constraints_multiple,
                                     np.logical_and(constraints_union,
                                                    constraints_child)))
            constraints_union = np.logical_or(constraints_union,
                                              constraints_child)
        for child_list_i, child_i in enumerate(parent_class.children):
            unique_constraints_mask = np.logical_and(constraints_of_children[child_list_i], np.logical_not(constraints_multiple))
            unique_constraints_list = np.nonzero(unique_constraints_mask)[0]
            num_unique_constraints = len(unique_constraints_list)
            if num_unique_constraints == 0:
                print("Warning: child {0} does not impose any unique constraints among the children of {1}".format(child_i, parent_i))
                continue
            bad_constraints = np.zeros(len(constraints), dtype=bool)
            descendants_of_child = get_descendant_classes(eq_classes, child_i)
            for desc_i in np.nonzero(np.logical_and(descendants_of_parent, np.logical_not(descendants_of_child)))[0]:
                if desc_i == parent_i:
                    continue
                num_bad_constraints = 0
                for constraint_i in unique_constraints_list:
                    if constraints[constraint_i].imposed_by[desc_i]:
                        num_bad_constraints += 1
                        bad_constraints[constraint_i] = True
                if num_bad_constraints:
                    # This actually occurs (in the example I saw, the constraint
                    # was between a different pair of nodes than the removed
                    # edge).
                    if verbose:
                        print("Warning: {0} of the {1} unique constraints imposed by child {2} of parent {3} are also imposed by class {4}, which descends from {3} but not from {2}".format(num_bad_constraints, num_unique_constraints, child_i, parent_i, desc_i))
                        print("Parent:")
                        print(eq_classes[parent_i])
                        print("Child:")
                        print(eq_classes[child_i])
                        print("Descendant:")
                        print(eq_classes[desc_i])
            identifying_constraint = None
            if sum(bad_constraints) == num_unique_constraints:
                print("Warning: *all* unique constraints imposed by child {0} of parent {1} are invalidated by some descendant of {1}".format(child_i, parent_i))
            else:
                identifying_constraint = np.nonzero(np.logical_and(unique_constraints_mask, np.logical_not(bad_constraints)))[0][0]
            children_identifying_constraint.append(identifying_constraint)
            if parent_class.children_identifying_constraint is not None:
                precomputed_constraint = parent_class.children_identifying_constraint[child_list_i]
                if not unique_constraints_mask[precomputed_constraint]:
                    print("Error: precomputed constraint is not unique")
                elif bad_constraints[precomputed_constraint]:
                    print("Error: precomputed constraint is bad")
            else:
                print("Error: no precomputed constraint")

        if write:
            parent_class.children_identifying_constraint = children_identifying_constraint
        print("\r{0}: {1:0.2f}%"
              .format(parent_i + 1,
                      100.0 * (parent_i + 1) / len(eq_classes)),
              end='', file=sys.stderr)
    print(file=sys.stderr)

def delete_operator_commutativity_TEST(d):
    # Test succeeded for d=4,5,6
    # Proof: Follows from ChickeringMeek2015 (SGES paper),
    # Lemma 2 ('The Deletion Lemma')
    solver = BBES(d, 0)
    equivalence_classes = solver.search_space['equivalence_classes']
    constraints = solver.search_space['constraints']
    for parent_i, parent_class in enumerate(equivalence_classes):
        print(parent_i, "/", len(equivalence_classes), end='\r')
        is_desc = get_descendant_classes(equivalence_classes, parent_i)
        for child_sub_i, child_i in enumerate(parent_class.children):
            child_constraint_i = parent_class.children_identifying_constraint[child_sub_i]
            child_constraint = constraints[child_constraint_i]
            for desc_i, desc_class in enumerate(equivalence_classes):
                if not is_desc[desc_i]:
                    continue
                if child_constraint.imposed_by[desc_i]:
                    # descendant of parent, and imposing id_constraint of child
                    for constraint_i, constraint in enumerate(constraints):
                        if constraint.imposed_by[child_i] and not constraint.imposed_by[desc_i]:
                            print("ERROR: for parent class")
                            print(parent_class)
                            print(", the constraint", constraint, "is imposed by child class")
                            print(child_class)   #unresovled reference child_class
                            print("but not by descendant class")
                            print(desc_class)
    print("delete_operator_commutativity_TEST DONE")

class BBES_state:
    """State in the BBES algorithm: set of equivalence classes"""
    def __init__(self, solver, superclass, classes_included=None,
                 required_connections=None):
        self.solver = solver # BBES class object using this state. TODO could be class variable if Solver is a singleton.
        self.superclass = superclass
        self.max_params = superclass.num_edges
        self.classes_included = classes_included
        if classes_included is not None:
            self.max_loglik = solver.compute_loglik(superclass.example_mB)
            self.min_params = self.compute_min_params_bruteforce() # requires max_params
        else:
            self.required_connections = required_connections
            self.max_loglik = None
            self.min_params = None
        self.key = None
        self.visited = False # is set to True when its children are compared
        #self.state_TEST()
    def is_singleton(self):
        return self.min_params == self.max_params
    def do_branch(self, child_id_constraint, child_superclass=None):
        # return a new state that imposes a constraint, and modify self to
        # not impose that constraint
        #child_superclass_i = self.superclass.children[child_i]
        #child_id_constraint = self.solver.search_space['constraints'][self.superclass.children_identifying_constraint[child_i]]
        if self.solver.without_precomp == 1:
            # without precomputed search space:
            child_state = BBES_state(self.solver,
                                     child_superclass,
                                     required_connections
                                     = self.required_connections[:]) # copy
            # set child_state's max_loglik and min_params
            child_state.max_loglik = (self.max_loglik
                                      - self.solver.compute_loglik_difference
                                        (child_id_constraint))
            child_state.min_params = self.min_params
            # modify self:
            self.required_connections.append(child_id_constraint)
            self.min_param_heuristic_update(child_id_constraint.v,
                                            child_id_constraint.w)
            self.key = None
        else:
            # with precomputed search space:
            child_state = BBES_state(self.solver,
                                     child_superclass,
                                     np.logical_and(self.classes_included,
                                                    child_id_constraint.imposed_by))
            # modify self:
            np.logical_and(self.classes_included,
                           np.logical_not(child_id_constraint.imposed_by),
                           out=self.classes_included)
            self.min_params = self.compute_min_params_bruteforce()
            self.key = None
            #self.state_test() # TO/DO comment out when done testing
            if self.solver.without_precomp == 2:
                child_state.required_connections = self.required_connections[:] # copy
                self_exact_min_params = self.min_params
                self.required_connections.append(child_id_constraint)
                self.min_param_heuristic_update(child_id_constraint.v,
                                                child_id_constraint.w)
                if self.min_params > self_exact_min_params:
                    print("ERROR: bound on min_params is incorrect in do_branch!")
        return child_state
    def min_param_heuristic_update(self, v, w):
        # a required connection between v and w was just added
        # (Attributes used here are initialized when a state is first visited)
        recompute = False
        self.num_del_ops_per_adj[v,w] -= 1
        self.num_del_ops_per_adj[w,v] -= 1
        if self.num_del_ops_per_adj[v,w] == 0:
            self.min_param_heuristic_update_advanced()
            recompute = True
        if self.comps[v] != self.comps[w]:
            self.comps.union(v, w)
            recompute = True
        if recompute:
            self.min_param_heuristic_recompute()
            #self.solver.compare_with_without_precomp_TEST(self, test_del_ops=False)
    def min_param_heuristic_update_advanced(self, verbose=False):
        #verbose=True
        # Updates self.required_edges and self.req_comps.
        d = self.solver.d
        original_cpdag = self.superclass.get_cpdag()
        if verbose:
            print(". original_cpdag:")
            print(original_cpdag)
        orig_skel = np.logical_or(original_cpdag, original_cpdag.T)
        orig_mU = np.logical_and(original_cpdag, original_cpdag.T)
        orig_mB = np.logical_and(original_cpdag, np.logical_not(orig_mU))
        req_adjs = np.logical_or(original_cpdag, original_cpdag.T)
        any_changes = False
        for v in range(d):
            for w in range(v):
                if self.num_del_ops_per_adj[v,w] > 0: # unused del ops on v,w
                    # delete operator(s) applicable to original cpdag
                    any_changes = True
                    req_adjs[v,w] = req_adjs[w,v] = False
                    # The immediate effect of the Delete operator may also
                    # include orienting some undirected edges to make
                    # v-structures; however, it doesn't include changes to
                    # the orientation of any directed edges.
        if verbose:
            print(". req_adjs after checking immediate delete ops:")
            print(req_adjs)
        while any_changes:
            any_changes = False
            fixed_orientations = (complete_pdag_fixed_orientations
                                  (original_cpdag, req_adjs))
            if verbose:
                print(". fixed_orientations:")
                print(fixed_orientations)
            for v in range(d):
                for w in range(v):
                    if not req_adjs[v,w]:
                        continue
                    x, y = v, w
                    if orig_mB[v,w]:
                        x, y = w, v
                    # Now x --> y or x --- y (required; possibly not fixed)
                    orig_Pa = (orig_mB[y,:]).copy()
                    orig_Pa[x] = False
                    # Additional requirements: either x --> y is fixed,
                    # or the adjacency between x and z is required (second
                    # option also with x and y switched).
                    sure_Pa = (np.logical_and
                               (np.logical_and(req_adjs[y,:],
                                               req_adjs[x,:]),
                                np.logical_or(fixed_orientations[y,:],
                                              fixed_orientations[x,:])))
                    if fixed_orientations[y,x]:
                        sure_Pa = (np.logical_or
                                   (sure_Pa,
                                    np.logical_and(fixed_orientations[y,:],
                                                   req_adjs[y,:])))
                    sure_Pa[x] = False
                    if verbose:
                        #print("  x, y = {0}, {1}; orig_Pa = {2}"
                        #      .format(x, y, orig_Pa))
                        print("  x, y = {0}, {1}; sure_Pa = {2}"
                              .format(x, y, sure_Pa))
                    if np.any(np.logical_and(sure_Pa, np.logical_not(orig_Pa))):
                        print("ERROR: sure_Pa not subset of orig_Pa")
                        0 / 0 # TODO: these actually occur
                    if np.any(np.logical_and(orig_Pa, np.logical_not(sure_Pa))):
                        # some node that was original a parent of {v,w} might
                        # not be a parent anymore: new delete operators
                        if verbose:
                            print("  ==> delete")
                        req_adjs[v,w] = req_adjs[w,v] = False
                        any_changes = True
                        continue
                    # What nodes z can become Pa or NA to {x,y} that aren't
                    # so originally? With fewer that 3 adjacencies, the only
                    # configuration in which z is Pa or NA is a v-structure,
                    # and by Chickering2002's Lemma 28, the only way to reach
                    # it from some other configuration is from 3 adjacencies.
                    # With 3 adjacencies, the only configurations in which z
                    # is *not* yet Pa or NA, are those where z is a child of
                    # both x and y.
                    might_become_Pa_or_NA = np.logical_and(orig_mB[:,y],
                                                           orig_mB[:,x])
                    # If both those arrows have fixed orientation, we can
                    # guarantee that the three nodes can't reach a Pa-or-NA
                    # configuration. Otherwise, we can't guarantee this.
                    both_fixed = np.logical_and(fixed_orientations[:,y],
                                                fixed_orientations[:,x])
                    pos_new_Pa_or_NA = np.logical_and(might_become_Pa_or_NA,
                                                      np.logical_not(both_fixed))
                    if verbose:
                        print("  x, y = {0}, {1}; pos_new_Pa_or_NA = {2}"
                              .format(x, y, pos_new_Pa_or_NA))
                    if np.any(pos_new_Pa_or_NA):
                        # some node that originally couldn't be in Pa or NA
                        # of {v,w} now might be: new delete operators
                        if verbose:
                            print("  ==> delete")
                        req_adjs[v,w] = req_adjs[w,v] = False
                        any_changes = True
            if verbose:
                print(". req_adjs after checking subsequent delete ops:")
                print(req_adjs)

        self.required_edges = req_adjs
        for v in range(d - 1):
            for w in range(1, d):
                if self.required_edges[v,w]:
                    self.req_comps.union(v, w)
    def min_param_heuristic_recompute(self):
        d = self.solver.d
        min_params = np.sum(self.required_edges) / 2
        for v in range(d):
            # Count number of unions in comps and req_comps. Req_comps will be
            # a refinement of comps. Add the difference the min_params.
            if self.comps[v] == v:
                min_params -= 1
            if self.req_comps[v] == v:
                min_params += 1
        self.min_params = min_params
    def compute_min_params_bruteforce(self):
        boundaries = self.solver.search_space['first_class_with_num_params']
        for num_params in range(len(boundaries) - 1):
            if np.any(self.classes_included[boundaries[num_params]:boundaries[num_params+1]]):
                return num_params
        # FIXED: numpy's nonzero() computes a list of all True's which will
        # often be slow. numpy 2.0 should have a function that finds the first
        # True. [Indeed, using np.any repeatedly gives tremendous speedup over
        # code below on n=6]
        #for i in range(self.superclass_index):
        #    if self.classes_included[i]:
        #        return self.solver.search_space['equivalence_classes'][i].num_edges
        #return self.max_params
    def state_TEST(self):
        # Test if there are any classes in the state which are not on the bottom
        # level yet do no have children within the state. If so, that would
        # make greedy search unusable as a way of determining min_params.
        # TEST FAILS: LOCAL MINIMA FOUND
        if False:
          for i in range(len(self.classes_included)):
            if not self.classes_included[i]:
                continue
            cur_class = self.solver.search_space['equivalence_classes'][i]
            if cur_class.num_edges == self.min_params:
                continue
            any_children_in_state = False
            for child_class_i in cur_class.children:
                if self.classes_included[child_class_i]:
                    any_children_in_state = True
                    break
            if not any_children_in_state:
                print("State contains a local minimum!")
                print("superclass:")
                print(self.superclass)
                print("local minimum:")
                print(cur_class)
                print("min_params:")
                print(self.min_params)

    def get_key(self):
        if self.key is None:
            # Queue takes the smallest key first
            self.key = self.solver.compute_score(self.max_loglik,
                                                 self.min_params)
        return self.key
    def __str__(self):
        # FIXME: won't work for precomputed case; have this call summary instead?
        ret = "superclass:\n"
        ret += self.superclass.__str__()
        if self.required_connections:
            ret += "\nrequired d-connections:"
            for req_conn in self.required_connections:
                ret += "\n * {0}".format(req_conn)
        else:
            ret += "\n no required d-connections"
        return ret
    def summary(self):
        if self.solver.search_space is not None:
            ret = "state[{0}".format(self.max_params)
            if self.min_params < self.max_params:
                boundaries = self.solver.search_space['first_class_with_num_params']
                if self.min_params < self.max_params - 1:
                    ret += (" ; {0}({1})"
                            .format(self.max_params - 1,
                                    np.sum(self.classes_included[boundaries[self.max_params-1]:boundaries[self.max_params]])))
                ret += (" ... {0}({1})"
                        .format(self.min_params,
                                np.sum(self.classes_included[boundaries[self.min_params]:boundaries[self.min_params+1]])))
            ret += "]: {0}".format(self.solver.score2display(self.get_key()))
            return ret
        else:
            ret = "state[{0}".format(self.max_params)
            if self.min_params < self.max_params:
                ret += " ... {0}".format(self.min_params)
            ret += "]: {0}".format(self.solver.score2display(self.get_key()))
            return ret


def draw_multivariate_data(true_B, true_O, N):
    # Same as in aelsem
    d = true_B.shape[0]
    m = np.zeros(d) # vector of means
    S = np.dot((np.linalg.inv(np.eye(d) - true_B)), np.dot(true_O, (np.linalg.inv(np.eye(d) - true_B)).T))
    Y = np.random.multivariate_normal(m, S, N) # Nxd
    return Y

def sample_covariance_matrix(X):
    # Input: X\in R^{Nxd}; Output: sample covariance matrix S
    return np.cov(X, rowvar=0, ddof=0)

def dag2cov(true_mB, N):
    # Special case of aelsem's model2cov with diagonal true_mO
    d = true_mB.shape[0]
    true_B = 1.0 * true_mB * np.random.standard_normal(true_mB.shape)
    true_O = np.random.standard_normal(true_mB.shape)
    true_O = math.sqrt(.1) * np.eye(d, dtype=bool) * np.dot(true_O, true_O.T)
    Y = draw_multivariate_data(true_B, true_O, N)
    return sample_covariance_matrix(Y)

def print_performance_row(header, data, num_general_measures):
    print("{0}:\t".format(header), end='')
    for i in range(num_general_measures):
        print("{0:0.1f}\t".format(data[i]), end='')
    for i in range(num_general_measures + 2, len(data)):
        print("{0:0.3f} ".format(data[i]), end='')
    print()


class BBES:
    """Solver for score-based DAG learning using branch&bound"""
    def __init__(self, d, without_precomp):
        self.d = d
        self.without_precomp = without_precomp
        if without_precomp != 1:
            self.search_space = search_space_prepare(d)
        else:
            self.search_space = None
    def score2display(self, score):
        return (score - self.score_display_offset) * self.score_display_factor
    def compute_clique_loglik(self, bin):
        mask = bin2mask(self.d, bin)
        if self.loglik_cache[bin] is not None:
            return self.loglik_cache[bin]
        if np.sum(mask) <= 1:
            self.loglik_cache[bin] = 0
            return 0
        # log likelihood maximized by Sigma [modelled] equal to sampleCov on
        # submatrix corresponding to clique. Then for this submatrix,
        # the trace term in the loglik vanishes. Compute the difference of the
        # remaining term with the case where all off-diag elts are 0.
        Ssub = self.sampleCov[mask,:][:,mask] # FIXME better indexing method? (also elsewhere)
        loglik_cluster = -.5 * self.N * np.log(np.linalg.det(Ssub))
        loglik_empty = -.5 * self.N * np.log(np.diag(Ssub).prod())
        self.loglik_cache[bin] = loglik_cluster - loglik_empty
        return self.loglik_cache[bin]
    def fraction_cluster_loglik_evals(self):
        evals_per_size = np.zeros(self.d + 1)
        total_per_size = np.zeros(self.d + 1)
        for bin in range(1 << self.d):
            s = np.sum(bin2mask(self.d, bin))
            total_per_size[s] += 1
            if self.loglik_cache[bin] is not None:
                evals_per_size[s] += 1
        return evals_per_size / total_per_size
    def compute_loglik_difference(self, identifying_constraint):
        v = identifying_constraint.v
        w = identifying_constraint.w
        S = identifying_constraint.S
        vbin = (1 << v)
        wbin = (1 << w)
        Sbin = mask2bin(S)
        loglik_difference = (self.compute_clique_loglik(Sbin)
                             - self.compute_clique_loglik(Sbin + vbin)
                             - self.compute_clique_loglik(Sbin + wbin)
                             + self.compute_clique_loglik(Sbin + vbin + wbin))
        return loglik_difference
    def compute_loglik(self, mB, verbose=False):
        # loglik_cache[bin]: log-likelihood of clique on set of nodes bin
        # (with bin in binary encoding)
        coefficients = collections.Counter()
        for v in range(self.d):
            parents = mB[v,:]
            bin_parents = mask2bin(parents)
            bin_parents_and_v = bin_parents + (1 << v)
            coefficients[bin_parents_and_v] += 1
            coefficients[bin_parents] -= 1
        loglik = 0
        if verbose:
            print("Result of compute_loglik for")
            print(mB)
        for bin, coefficient in coefficients.items():
            if coefficient == 0:
                continue
            clique_loglik = self.compute_clique_loglik(bin)
            if verbose:
                print("  {0} x {1}: {2}".format(coefficient, bin,
                                                clique_loglik))
            loglik += coefficient * clique_loglik
        if verbose:
            print("  = {0}".format(loglik))
        return loglik

    def compute_score(self, loglik, num_params):
        #return -(2 * loglik - 2 * num_params)
        return -(loglik - 1 * self.penalty_weight * num_params)


    def compute_score_AIC(self, loglik, num_params):
        # return -(loglik - 1* self.penalty_weight * num_params)
        return -(2 * loglik - 2 * num_params)


    def consider_solution(self, solution_class, loglik, verbose=False):
        #solution_class = (self.search_space['equivalence_classes']
        #                  [solution_class_index])
        #loglik = self.compute_loglik(solution_class.example_mB)
        params = solution_class.num_edges
        score = self.compute_score(loglik, params)
        # Again, smaller is better
        if score < self.best_class_score:
            self.best_class = solution_class
            self.best_class_score = score # TODO: implement top-k / window
            self.top_threshold = score
            if verbose:
                print("Found an improved best equivalence class so far (score {0}):"
                      .format(self.score2display(score)))
                print(solution_class)
    def enqueue(self, state):
        key = state.get_key()
        if key < self.top_threshold + 1e-9:
            self.Q.put((key, state))
    def compute_delete_operators(self, current_state, current_cpdag, verbose):
        if verbose >= 3:
            print("Computing delete operators for state:")
            print(current_state)
            print(current_cpdag)
        valid_delete_operators = generate_valid_delete_operators(current_cpdag)
        if verbose >= 3:
            print("Valid delete operators:")
            for del_op in valid_delete_operators:
                print(del_op)
        current_state.available_del_ops = []
        current_state.del_ops_used = 0
        current_state.num_del_ops_per_adj = np.zeros((self.d, self.d))
        #current_state.required_edges = np.logical_or(current_cpdag, current_cpdag.T)
        current_state.comps = UnionFind()
        current_state.req_comps = UnionFind()
        for del_op in valid_delete_operators:
            if del_op in current_state.required_connections:
                # del_op is explicitly in list of disallowed
                # separations
                if verbose >= 3:
                    print("{0}: explicitly forbidden".format(del_op))
                continue
            child_pdag = apply_delete_operator(current_cpdag, del_op)
            child_cpdag = complete_pdag(child_pdag)
            conflict = False
            for req_conn in current_state.required_connections:
                if is_d_separated(child_cpdag,
                                  req_conn.v, req_conn.w,
                                  req_conn.S):
                    # del_op would lead to a disallowed separation
                    if verbose >= 3:
                        print("{0}: conflicts with {1}"
                              .format(del_op, req_conn))
                    conflict = True
                    break
            if not conflict:
                v = del_op.v
                w = del_op.w
                current_state.num_del_ops_per_adj[v,w] += 1
                current_state.num_del_ops_per_adj[w,v] += 1
                #current_state.required_edges[v,w] = False
                #current_state.required_edges[w,v] = False
                child_class = EquivalenceClass(child_cpdag)
                child_class.is_pdag_completed = True
                child_loglik = (current_state.max_loglik
                                - self.compute_loglik_difference
                                  (del_op))

                self.consider_solution(child_class,
                                       child_loglik,
                                       verbose>=1)
                if verbose >= 3:
                    print("{0}: available; loglik {1}".format(del_op,
                                                              child_loglik))

                key = ()
                for heur in self.branching_heuristic:
                    if heur == "1":
                        key_here = np.random.rand()
                    elif heur == "2":
                        key_here = (1 if child_loglik > -self.top_threshold
                                else 0)
                    elif heur == '3':
                        key_here = min(child_loglik,
                                   -self.top_threshold)
                    elif heur == '4':
                        key_here = child_loglik
                    elif heur == '5':
                        key_here = (child_loglik if child_loglik > -self.top_threshold
                                else 0)
                    elif heur == '6':
                        key_here = -child_loglik
                    else:
                        raise ValueError("Unknown branching heuristic")
                    key = key + (key_here, )

                current_state.available_del_ops.append((key, del_op))

        current_state.available_del_ops.sort() # maxdep heuristic

        for req_conn in current_state.required_connections:
            current_state.comps.union(req_conn.v, req_conn.w)
        # Update current_state.required_edges and .req_comps.union
        current_state.min_param_heuristic_update_advanced()
        current_state.min_param_heuristic_recompute()
        if len(current_state.available_del_ops) == 0:
            # state turns out to be a singleton
            current_state.min_params = current_state.max_params
    def compare_with_without_precomp_TEST(self, current_state,
                                          test_del_ops=True):
        current_superclass = current_state.superclass
        any_errors = False
        critical_error = False

        if test_del_ops:
            # Test if list of available delete operators is correct
            actual_del_ops = set()
            actual_del_ops_found = set()
            for child_i, child_class_index in enumerate(current_superclass.children):
                if not current_state.classes_included[child_class_index]:
                    continue
                child_constraint = self.search_space['constraints'][current_superclass.children_identifying_constraint[child_i]]
                actual_del_ops.add((child_constraint.v, child_constraint.w,
                                    mask2bin(child_constraint.S)))
            if len(current_state.available_del_ops) != len(actual_del_ops):
                print("COMPARE: number of del_ops found ({0}) != actual number ({1})".format(len(current_state.available_del_ops), len(actual_del_ops)))
                any_errors = True
            for del_op_pair in current_state.available_del_ops:
                del_op = del_op_pair[1]
                if del_op.v < del_op.w:
                    del_op_normalized = (del_op.v, del_op.w, mask2bin(del_op.S))
                else:
                    del_op_normalized = (del_op.w, del_op.v, mask2bin(del_op.S))
                if del_op_normalized in actual_del_ops:
                    actual_del_ops_found.add(del_op_normalized)
                else:
                    print("COMPARE: incorrect del_op found: {0}"
                          .format(del_op))
                    any_errors = True

            if any_errors:
                # WAS: for del_op in actual_del_ops:
                for child_i, child_class_index in enumerate(current_superclass.children):
                    if not current_state.classes_included[child_class_index]:
                        continue
                    child_constraint = self.search_space['constraints'][current_superclass.children_identifying_constraint[child_i]]
                    del_op = (child_constraint.v, child_constraint.w,
                              mask2bin(child_constraint.S))
                    #print(child_constraint)
                    #print(del_op)
                    #print(bin2mask(self.d, del_op[2]))
                    print("COMPARE: correct del_op: {0}{1}; resulting superclass:"
                          .format(Constraint(del_op[0], del_op[1],
                                             bin2mask(self.d, del_op[2])),
                                  "" if del_op in actual_del_ops_found
                                  else " (not found!)"))
                    print(self.search_space['equivalence_classes'][child_class_index])
                any_errros = True

        # Check that edges are marked as required iff they really are required
        actual_required_edges = np.logical_or(current_superclass.example_mB,
                                              current_superclass.example_mB.T)
        for eq_class_i in range(len(current_state.classes_included)):
            if not current_state.classes_included[eq_class_i]:
                continue
            eq_class = self.search_space['equivalence_classes'][eq_class_i]
            skel = np.logical_or(eq_class.example_mB, eq_class.example_mB.T)
            actual_required_edges = (np.logical_and(actual_required_edges,
                                                    skel))
            if np.any(np.logical_and(current_state.required_edges,
                                     np.logical_not(skel))):
                print("COMPARE: ERROR: edge classified as required is missing from this class:")
                print(eq_class)
                any_errors = True
                critical_error = True
        # Suppressing part of following for now: I know the bound is not tight.
        #if np.any(current_state.required_edges != actual_required_edges):
        if np.any(np.logical_and(current_state.required_edges, np.logical_not(actual_required_edges))):
            print("COMPARE: required_edges is different from actual:")
            print(current_state.required_edges)
            print("; should be:")
            print(actual_required_edges)
            print("Log for computing required_edges:")
            current_state.min_param_heuristic_update_advanced(verbose=True)
            any_errors = True

        if any_errors:
            print("COMPARE: above messages were found for the following state:")
            print(current_state)
            if critical_error:
                0 / 0
            print("COMPARE: done")


    def branching_heuristic_with_oracle(self, current_state, current_superclass, verbose):
        if verbose >= 3 and not current_state.visited:
            print("Expanding state for the first time:")
            print(current_superclass)
            print("delta:\tbotrem:\t|S|:\tconstraint:")
        best_split = ((float('-inf'), ), None)
        # compute number of classes currently in bottom level
        boundaries = self.search_space['first_class_with_num_params']
        bd_lo = boundaries[current_state.min_params]
        bd_hi = boundaries[current_state.min_params+1]
        bot_old = np.sum(current_state.classes_included[bd_lo:bd_hi])

        '''
        bd_down = bd_lo
        bd_top = boundaries[current_state.max_params]
        bd_top_child = boundaries[current_state.max_params-1]
        '''
        # num_class = np.sum(current_state.classes_included[bd_down:bd_top])

        #print(num_class/bot_old)
        for child_i, child_class_index in enumerate(current_superclass.children):
            if not current_state.classes_included[child_class_index]:
                continue
            child_loglik = self.compute_loglik(self.search_space['equivalence_classes'][child_class_index].example_mB)
            delta = current_state.max_loglik - child_loglik
            likelratio = delta / current_state.max_loglik
            score = self.compute_score(child_loglik,
                                       current_state.min_params)

            score_AIC = self.compute_score_AIC(child_loglik,
                                       current_state.min_params)
            score_2 = 2*self.penalty_weight * current_state.min_params - child_loglik
            child_constraint = self.search_space['constraints'][current_superclass.children_identifying_constraint[child_i]]
            cond_set_size = np.sum(child_constraint.S)
            # Compute number of classes remaining in current state
            # at original bottom lvl
            bot_rem = np.sum(np.logical_and
                             (current_state.classes_included
                              [bd_lo:bd_hi],
                              np.logical_not(child_constraint.imposed_by
                                             [bd_lo:bd_hi])))
            AIC = child_loglik * -2 * self.N +2 * (current_state.max_params-1)/self.N
            # child_class = np.sum(child_constraint.imposed_by[bd_down:bd_top_child])
            if verbose >= 3 and not current_state.visited:
                # print quantities for all children
                print("{0:0.3f}{1}\t{2}\t{3}\t{4}"
                      .format(delta / self.penalty_weight,
                              '*' if score > self.top_threshold
                              else '',
                              bot_rem, cond_set_size, child_constraint))
            key = ()
            for heur in self.branching_heuristic:
                if heur == 'arbitrary':
                    # no heuristic: process children in order given
                    key_here = 0
                elif heur == 'random':
                    key_here = np.random.rand()
                elif heur == 'maxabs':
                    key_here = abs(delta - self.penalty_weight)
                elif heur == 'maxdep':
                    # dependencies first, then indeps starting with
                    # weakest
                    key_here = delta
                elif heur == 'depfirst':
                    # deps first, then indeps starting with strongest
                    key_here = (delta if delta > self.penalty_weight
                                else -delta)
                elif heur == 'mindep':
                    key_here = -delta
                elif heur == 'maxdep_maxed':
                    # If likelihood drop would make score worse than
                    # threshold, precise drop doesn't matter, so compare
                    # those as equal.
                    # Note: if child state doesn't seem strong, but
                    # cuts off part of the bottom so that its
                    # min_params increases by one or more, the child
                    # state will be discardable after all.
                    key_here = min(score,
                                   self.top_threshold)
                elif heur == 'maxdep_maxed_fracbot':
                    # DEPRECATED: prefer ('strongdep', _nojump)
                    # a score that crosses the threshold is always
                    # treated as better than a score that doesn't,
                    # regardless of bot_rem
                    key_here = ((score if score <= self.top_threshold
                                 else (self.top_threshold
                                       + 10 * self.penalty_weight))
                                + (self.penalty_weight
                                   * (bot_old - bot_rem) / bot_old))
                elif heur == 'maxdep_maxed_fracbot_nojump':
                    key_here = (min(score, self.top_threshold)
                                + (self.penalty_weight
                                   * (bot_old - bot_rem) / bot_old))
                elif heur == 'maxdep_maxed_fracbot_nojump2':
                    key_here = (min(score, self.top_threshold)
                                + (2 * self.penalty_weight
                                   * (bot_old - bot_rem) / bot_old))
                elif heur == 'strongdep':
                    key_here = (1 if score > self.top_threshold
                                else 0)
                elif heur == 'maxdep_maxed_int':
                    # [used to have int() instead of math.floor()]
                    # (int rounds towards 0; not sure if these values
                    # are positive or negative, but as long as they all
                    # have the same sign, it doesn't matter)
                    key_here = math.floor(min(score, self.top_threshold)
                                          / self.penalty_weight)
                elif heur == 'maxdep_maxed_int_new':
                    # use floor rather than int; put distance between
                    # strong deps and rest
                    key_here = math.floor((score if
                                           score <= self.top_threshold
                                           else (self.top_threshold
                                                 + 10
                                                 * self.penalty_weight))
                                          / self.penalty_weight)
                elif heur == 'maxdep_maxed_dropint':
                    # strong dep: 0; otherwise < 0
                    key_here = math.floor(min(score
                                              - self.top_threshold, 0)
                                          / self.penalty_weight)
                elif heur == 'smallk':
                    key_here = -cond_set_size
                elif heur == 'largek':
                    key_here = cond_set_size
                elif heur == 'smallbot':
                    key_here = -bot_rem
                elif heur == 'maxdep_fracbot':
                    key_here = (delta + self.penalty_weight
                                * (bot_old - bot_rem) / bot_old)

                #different heuristics for different states(size different)
                elif heur == '1':
                    if (current_state.max_params - current_state.min_params) > (current_state.max_params / 2):
                        key_here = delta + 2 * (bot_old - bot_rem) / bot_old
                    else:
                        key_here = score
                # AIC score
                elif heur == '2':
                    key_here = score

                #higher penalty
                elif heur =='3':
                    score_2 = 1.5 * self.penalty_weight * current_state.min_params - child_loglik
                    key_here = score_2

                elif heur =='4':
                    key_here = (min(score_AIC, self.top_threshold)
                     + (2 * self.penalty_weight
                        * (bot_old - bot_rem) / bot_old))

                elif heur =='5':
                    key_here = (min(score, self.top_threshold)
                     + (2 * self.penalty_weight
                        * (bot_old - bot_rem) / bot_old))

                elif heur =='6':
                    key_here = (min(score_AIC, self.top_threshold)
                     + (self.penalty_weight(bot_old - bot_rem) / bot_old))

                else:
                    raise ValueError("Unknown branching heuristic")
                key = key + (key_here, )
            if key > best_split[0]:
                best_split = (key, child_i)

        return best_split[1]
    
    def main_loop(self, max_iterations=-1, verbose=0, show_progress=True):
        current_state = None
        while True:
            self.num_iterations += 1
            if self.num_iterations == max_iterations:
                if verbose >= 1:
                    print("Reached branch & bound iteration limit of {0}"
                          .format(max_iterations))
                break
            if verbose >= 2:
                print("  === ITERATION {0} ===".format(self.num_iterations))

            if current_state is None:
                # best-first search (else continue with state from previous
                # iteration for depth-first search)
                try:
                    (current_key, current_state) = self.Q.get(block=False)
                    self.optimistic_score = current_key
                except Queue.Empty:
                    self.optimistic_score = float("inf")
                    if verbose >= 2:
                        print("DONE: Queue is empty")
                    break
            else:
                current_key = None # FIXME: compute this for dfs
            if verbose >= 2:
                print("CURRENT STATE", current_state.summary())
                #print(current_state.classes_included)
            if show_progress:
                print("\roptimistic={0:0.4f}\tbest={1:0.4f}\tthreshold={2:0.4f} "
                      .format(self.score2display(self.optimistic_score),
                              self.score2display(self.best_class_score),
                              self.score2display(self.top_threshold)),
                      end='', file=sys.stderr)

            if current_key > self.top_threshold + 1e-9:
                # FIXME: while doing dfs, this should just break out of that dfs
                if verbose >= 2:
                    print("DONE: States left in the queue are worse than known top solutions")
                break

            if not current_state.is_singleton():
                current_superclass = current_state.superclass
                if self.without_precomp != 1: # WAS: == 0
                    # Using precomputed search space
                    if self.without_precomp == 2:
                        # Also do updates for the no-precomputation case
                        if not hasattr(current_state, 'available_del_ops'):
                            current_cpdag = current_superclass.get_cpdag()
                            self.compute_delete_operators(current_state, current_cpdag, verbose)
                            self.compare_with_without_precomp_TEST(current_state)
                    child_i = self.branching_heuristic_with_oracle(current_state, current_superclass, verbose)
                    if verbose >= 2:
                        print("Branching on constraint", self.search_space['constraints'][current_superclass.children_identifying_constraint[child_i]])
                        #print(self.search_space['constraints'][current_superclass.children_identifying_constraint[child_i]].imposed_by)
                    child_constraint = self.search_space['constraints'][current_superclass.children_identifying_constraint[child_i]]
                    child_superclass = self.search_space['equivalence_classes'][current_superclass.children[child_i]]
                    child_state = current_state.do_branch(child_constraint,
                                                          child_superclass)
                    self.num_branches += 1
                    if not current_state.visited:
                        current_state.visited = True
                        self.num_visited += 1
                    # Considering a child as a solution could be done already
                    # when that child's loglik is computed. Not doing that in
                    # the precomputed case, because there all logliks are
                    # already computed on the first expansion, which might be
                    # unrealistic.
                    self.consider_solution(child_state.superclass,
                                           child_state.max_loglik, verbose>=1)
                    if self.use_peek:
                        # min_params may have changed, so check again
                        boundaries = self.search_space['first_class_with_num_params']
                        bd_lo = boundaries[current_state.min_params]
                        bd_hi = boundaries[current_state.min_params+1]
                        first_class_i = bd_lo + np.nonzero(current_state.classes_included[bd_lo:bd_hi])[0][0]
                        solution_class = self.search_space['equivalence_classes'][first_class_i]
                        self.consider_solution(solution_class,
                                               self.compute_loglik(solution_class.example_mB), verbose>=1)
                    self.enqueue(current_state)
                    self.enqueue(child_state)
                    if verbose >= 2:
                        print("Enqueued two states:")
                        print("*", current_state.summary())
                        print("*", child_state.summary())


                else:
                    # Case of no precomputed search space:
                    # Determine best branch
                    current_cpdag = current_superclass.get_cpdag()
                    if not hasattr(current_state, 'available_del_ops'):
                        self.compute_delete_operators(current_state, current_cpdag, verbose)
                        # min_params may have increased, reducing this state's
                        # priority: put it back in the queue and pick a new one
                        self.enqueue(current_state)
                        current_state = None
                        continue
                    # TODO: implement better branching heuristic for this case

                    if current_state.del_ops_used == len(current_state.available_del_ops):
                        # this should be redundant for a sufficiently good
                        # min_param heuristic
                        current_state.min_params = current_state.max_params
                        continue

                    child_constraint = current_state.available_del_ops[current_state.del_ops_used][1]

                    current_state.del_ops_used += 1

                    if current_state.del_ops_used == len(current_state.available_del_ops):
                        # this should be redundant for a sufficiently good
                        # min_param heuristic
                        current_state.min_params = current_state.max_params
                    pdag = apply_delete_operator(current_cpdag,
                                                 child_constraint)
                    child_superclass = EquivalenceClass(pdag)
                    if verbose >= 2:
                        print("Branching on constraint", child_constraint)
                    child_state = current_state.do_branch(child_constraint,
                                                          child_superclass)
                    self.num_branches += 1
                    if not current_state.visited:
                        current_state.visited = True
                        self.num_visited += 1
                    self.enqueue(current_state)
                    self.enqueue(child_state)
                    if verbose >= 2:
                        print("Enqueued two states:")
                        print("*", current_state.summary())
                        print("*", child_state.summary())
                # regardless of whether search space is precomputed:
            current_state = None

    def solve(self, sampleCov, N,
              branching_heuristic=None, use_peek=False,
              verbose=1, show_progress=True):
        # Verbosity levels:
        # -1: silent, and don't even output answer
        # 0: silent (just progress indicator on stderr if show_progress is True)
        # 1: report improved solutions
        # 2: report what happens in each iteration
        if self.d != sampleCov.shape[0]:
            raise ValueError("BBES solver initialized to dimension {0} but called to solve on dimension {1}".format(self.d, sampleCov.shape[0]))
        # Initialize members specific to the data set:
        self.sampleCov = sampleCov
        self.N = N
        self.penalty_weight = .5 * np.log(self.N)
        self.loglik_cache = (1 << self.d)*[None]
        # Use a linear transformation to make scores easier to interpret:
        # 0 = score of saturated model; +1: remove one param (so higher=better)
        self.score_display_offset = (self.compute_score
                                     (self.compute_clique_loglik((1 << self.d)
                                                                 - 1),
                                      self.d * (self.d - 1) / 2))
        self.score_display_factor = -1.0 / self.penalty_weight
        # Initialize algorithm settings
        self.branching_heuristic = branching_heuristic
        self.use_peek = use_peek
        # Initialize performance measures
        self.num_iterations = -1
        self.num_branches = 0
        self.num_visited = 0
        # Initialize queue and best solutions
        self.optimistic_score = float("-inf")
        self.best_class = None
        self.best_class_score = float("inf")
        self.top_threshold = float("inf") # Similar to best_class_score, but different if we're looking for the top k solutions, or some window below the best score
        self.Q = Queue.PriorityQueue() # of (key, BBES_state); smallest key first
        if self.without_precomp == 1:
            current_state = BBES_state(self,
                                       saturated_class(self.d),
                                       required_connections=[])
            example_mB = np.tril(np.ones((self.d, self.d), dtype=bool), k=-1)
            current_state.max_loglik = self.compute_loglik(example_mB)
            current_state.min_params = 0
        else:
            num_eq_classes = len(self.search_space['equivalence_classes'])
            current_state = BBES_state(self,
                                       self.search_space['equivalence_classes'][-1],
                                       np.ones(num_eq_classes, dtype=bool))
            if self.without_precomp == 2:
                current_state.required_connections = []
        self.optimistic_score = current_state.get_key()
        self.consider_solution(current_state.superclass,
                               current_state.max_loglik, verbose>=1)
        self.Q.put((current_state.get_key(), current_state))
        # FIXME: Removed try-except because I want to abort entire tests
        #try:
        self.main_loop(verbose=verbose, show_progress=show_progress)
        #except KeyboardInterrupt:
        #    print("Exectution aborted by user")

        if verbose >= 0:
            print("Results:")
            if self.optimistic_score > self.top_threshold + 1e-9:
                # B&B was run to completion: results are guaranteed to be
                # correct.
                # self.optimistic_score may be inaccurate in this case, as
                # states that score worse than the threshold are not enqueued.
                pass
            else:
                print("Algorithm was not run to completion, so results may not be optimal.")
                print("Bound on remaining scores: {0}".format(self.score2display(self.optimistic_score)))
            print("Top equivalence class(es) found:")
            print(self.best_class)
            print("with score {0}".format(self.score2display(self.best_class_score)))

def run_experiment(d, p_edge, num_repeats=100, N=10000,
                   without_precomp=1,
                   branching_heuristic=None, use_peek=False,
                   solver=None, verbose=-1, show_progress=True):   #changed verbose from -1 to 2
    if solver is None:
        solver = BBES(d, without_precomp)
    search_space = solver.search_space # for uniformly sampling an eq class
    #for constr in search_space['constraints']:
    #    print(constr)
    if False:
        if d < 6: # extra safeguard: testing for d==6 takes days
            unique_constraints_TEST(search_space['equivalence_classes'],
                                    search_space['constraints'])

    num_general_measures = 3
    performance_results = np.zeros((num_repeats, num_general_measures + d+1))
    total_time = 0
    for repeat in range(num_repeats):
        np.random.seed(repeat)
        if False:
            generating_class_i = np.random.randint(0, len(search_space['equivalence_classes']))
            #generating_class_i = 0
            generating_mB = (search_space['equivalence_classes'][generating_class_i]
                             .example_mB)
        else:
            # tril(A, -1): zero out A on and above main diagonal
            generating_mB = np.tril(np.random.rand(d, d) < p_edge, k=-1)
            pi = np.random.permutation(d)
            generating_mB = generating_mB[pi,:][:,pi]
        sampleCov = dag2cov(generating_mB, N)

        #print("Calling solver.solve()", file=sys.stderr)
        start_time = time.time()
        solver.solve(sampleCov, N,
                     branching_heuristic=branching_heuristic, use_peek=use_peek,
                     verbose=verbose, show_progress=show_progress)
        time_here = time.time() - start_time
        total_time += time_here
        if verbose >= 1:
            print("Time spent:", time_here, "sec")
            print("Generating model was")
            print(generating_mB)
            print("with score {0}".format
                  (solver.score2display
                   (solver.compute_score(solver.compute_loglik(generating_mB),
                                         np.sum(generating_mB)))))
        if show_progress:
            print("\trepeat {0} done".format(repeat),
                  end='', file=sys.stderr)
        performance_results[repeat, 0] = solver.num_iterations
        performance_results[repeat, 1] = solver.num_branches
        performance_results[repeat, 2] = solver.num_visited
        performance_results[repeat, num_general_measures:] = solver.fraction_cluster_loglik_evals()
    if show_progress:
        print(file=sys.stderr)
    print("Performance results:")
    print("Simulator settings: repeats={2}, d={0}, N={3}, p_edge={1}"
          .format(d, p_edge, num_repeats, N))
    print("\t#iter\t#branch\t#visit\t/loglik_evals per cluster size >=2")
    mean = np.mean(performance_results, axis=0)
    std = np.std(performance_results, axis=0)
    median = np.median(performance_results, axis=0)
    #np.percentile(performance_results, q, axis=0) # q=0,50,100=min,median,max
    print_performance_row("mean", mean, num_general_measures)
    print_performance_row("std ", std, num_general_measures)
    print_performance_row("q 50", median, num_general_measures)
    print("Total time:", total_time, "sec")
    #loglik_test(solver, d, sampleCov, N)
    return mean, median

def evaluate_heuristic(heuristic_i, without_precomp, big):
    #is_d_separated_TEST()

    #branching_heuristic = ('random', ) #('arbitrary', )
    #branching_heuristic = ('maxabs', )
    #branching_heuristic = ('maxdep', )
    #branching_heuristic = ('depfirst', )
    #branching_heuristic = ('mindep', )
    #branching_heuristic = ('smallk', '')
    #branching_heuristic = ('largek', '')
    #branching_heuristic = ('smallk', 'maxdep')
    #branching_heuristic = ('smallbot', 'maxdep')
    #branching_heuristic = ('smallbot', 'maxdep_maxed', 'smallk')
    #branching_heuristic = ('maxdep_fracbot', )
    #branching_heuristic = ('maxdep_maxed', )
    #branching_heuristic = ('maxdep_maxed', 'smallbot', 'smallk')
    #branching_heuristic = ('maxdep_maxed_fracbot_nojump', 'smallk')
    #branching_heuristic = ('maxdep_maxed_fracbot_nojump2', 'smallk')
    #branching_heuristic = ('strongdep', 'smallbot', 'smallk', 'maxdep')
    #branching_heuristic = ('maxdep_maxed_int', 'smallbot', 'smallk', 'maxdep')
    #branching_heuristic = ('maxdep_maxed_int', 'smallbot', 'smallk', 'maxdep_maxed')
    #branching_heuristic = ('maxdep_maxed_int', 'smallbot', 'maxdep')
    #branching_heuristic = ('maxdep_maxed_dropint', 'smallbot', 'maxdep')

    heuristic_list = [('random',),
                      ('maxabs',),
                      ('maxdep',),
                      ('depfirst',),
                      ('mindep',),
                      ('cool',),
                      # multiple measures:
                      ('smallk', 'maxdep'), # or something more realistic?
                      ('largek', 'maxdep'), # or something more realistic?
                      ('smallbot', 'maxdep'),
                      # care about threshold:
                      ('smallbot', 'maxdep_maxed', 'smallk'),
                      ('maxdep_maxed_int', 'smallbot', 'maxdep'),
                      ('maxdep_maxed_fracbot_nojump', ),
                      ('maxdep_maxed_fracbot_nojump2', ),
    ]
    if heuristic_i < len(heuristic_list):
        # heuristic_i 0..11: all branching heuristics, without peeking
        heuristic_list = [heuristic_list[heuristic_i]]
        use_peek = False
    else:
        # heuristic_i 12..15: final four branching heuristics with peeking
        heuristic_list = [heuristic_list[heuristic_i-4]]
        use_peek = True
    #heuristic_list = [branching_heuristic]

    #use_peek = False

    if big == 4:
        d = 6
        Ns = [100, 1000, 10000]
        ps = [.05, .1, .15, .2, .25, .3, .4, .5, .65, .8]
    if big == 3:
        d = 6
        Ns = [100, 1000, 10000]
        ps = [.2, .4, .6, .8]
    elif big == 2:
        # used for tables in PGM paper
        d = 6
        Ns = [100, 10000]
        ps = [.2, .4, .8]
    elif big == 1:
        d = 5
        Ns = [100, 10000]
        ps = [.2, .4, .8]
    elif big == 0:
        d = 4
        Ns = [100, 10000]
        ps = [.2, .4, .8]
    else: # -1
        d = 3
        Ns = [100, 10000]
        ps = [.2, .8]
    if without_precomp:
        raise ValueError("Trying to evaluate performance using without_precomp")
    for branching_heuristic in heuristic_list:
        eval_performance(d, Ns, ps, branching_heuristic, use_peek, without_precomp)

def eval_performance(d, Ns, ps, branching_heuristic, use_peek, without_precomp):
    mean_num_branches = np.zeros((len(Ns), len(ps)))
    mean_num_visits = np.zeros((len(Ns), len(ps)))
    solver = BBES(d, without_precomp) # loading precomputed data takes several minutes for d=6, so do it just once for all datasets
    for i, N in enumerate(Ns):
        print("*** N = {0} ***".format(N))
        for j, p_edge in enumerate(ps):
            mean, median = run_experiment(d, p_edge, num_repeats=100, N=N,
                                          branching_heuristic=branching_heuristic,
                                          use_peek=use_peek,
                                          solver=solver, show_progress=True)
            #print()
            mean_num_branches[i,j] = mean[1]
            mean_num_visits[i,j] = mean[2]
        print()
    alg_desc = "{0}{1}".format(branching_heuristic,
                                "; use_peek" if use_peek else "")
    print("mean #branches: (d={0}; branching_heuristic={1})\np \\ N"
          .format(d, alg_desc, end=''))
    for i, N in enumerate(Ns):
        print("\t{0}".format(N), end='')
    print()
    for j, p_edge in enumerate(ps):
        print(p_edge, end='')
        for i, N in enumerate(Ns):
            print("\t{0:0.1f}".format(mean_num_branches[i,j]), end='')
        print()
    #
    print("mean #visits: (d={0}; branching_heuristic={1})\np \\ N"
          .format(d, alg_desc, end=''))
    for i, N in enumerate(Ns):
        print("\t{0}".format(N), end='')
    print()
    for j, p_edge in enumerate(ps):
        print(p_edge, end='')
        for i, N in enumerate(Ns):
            print("\t{0:0.2f}".format(mean_num_visits[i,j]), end='')
        print()
    #
    # LaTeX output format " & $000.00$ & $700.50$ & $000.00$ & $700.50$ \\ %..."
    for i, N in enumerate(Ns):
        for j, p_edge in enumerate(ps):
            print(" & ${0:0.1f}$".format(mean_num_branches[i,j]), end='')
    print(" \\\\ % #branches; {0}".format(alg_desc))
    for i, N in enumerate(Ns):
        for j, p_edge in enumerate(ps):
            print(" & ${0:0.2f}$".format(mean_num_visits[i,j]), end='')
    print(" \\\\ % #visits; {0}".format(alg_desc))

def main(heuristic_i, big):
    print("Heuristic number:", heuristic_i)
    print("Experiment size:", big)
    evaluate_heuristic(heuristic_i, 0, big)

if __name__ == "__main__":
    #generate_delete_operators_TEST()
    #lemma_34_TEST(6)
    #delete_operator_commutativity_TEST(6)
    if True:
        '''
        # Run a set of experiments to get performance statistics.
        # First command line argument: heuristic number 0-15
        if len(sys.argv) >= 2:
            heuristic_i = int(sys.argv[1])
        else:
            heuristic_i = 5# heuristic $s$ (maxdep): simple and pretty fast
        # Second command line argument determines size & number of experiments;
        # use 2 for PGM
        if len(sys.argv) >= 3:
            big = int(sys.argv[2])
        else:
            big = 2# use 5 instead of 6 variables by default
        main(heuristic_i, big)
        '''

        # run coding task here
        # heuristic number 1-6
        run_experiment(5, p_edge=.5, num_repeats=100, N=10000,
                       without_precomp=1,  use_peek=False,
                       branching_heuristic=('5'),
                       verbose=-1, show_progress=(-1 <= 1))

    else:
        # Run a single custom experiment (or multiple repeats of the same one).
        # without_precomp = 0: precomp; 1: normal; 2: normal with precomp tests
        without_precomp = 2
        verbose = 2
        run_experiment(5, p_edge=.5, num_repeats=1, N=10000,
                       without_precomp=without_precomp,
                       branching_heuristic=('maxdep_maxed', 'smallbot', 'smallk'), use_peek=False,
                       verbose=verbose, show_progress=(verbose<=1))
