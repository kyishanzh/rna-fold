#!/usr/bin/env /usr/bin/python
import argparse
import numpy as np

from pyrosetta import *
import concurrent.futures
from pyrosetta.rosetta import *
from pyrosetta.rosetta.core.pose.rna import *
from pyrosetta.rosetta.core.scoring import ScoreFunction, ScoreType
from pyrosetta.rosetta.protocols.constraint_movers import ConstraintSetMover
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.pose import *
import re, sys
import os, shutil
import math
import random


def fold_from_cst(args):
    global rna_lowres_sf
    init(
        '-mute all -hb_cen_soft  -relax:dualspace true -relax:default_repeats 3 -default_max_cycles 200 -detect_disulf -detect_disulf_tolerance 3.0')

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    rna_lowres_sf = core.scoring.ScoreFunctionFactory.create_score_function("rna/denovo/rna_lores_with_rnp_aug.wts")

    global op_score
    seq = read_fasta(args.FASTA).replace('T', 'U')
    args.seq = seq.lower()

    op_score = create_score_function('ref2015')
    op_score.set_weight(rosetta.core.scoring.atom_pair_constraint, 9.0)
    op_score.set_weight(rosetta.core.scoring.dihedral_constraint, 4.0)
    op_score.set_weight(rosetta.core.scoring.angle_constraint, 4.0)

    op_score.set_weight(rosetta.core.scoring.fa_rep, 9.0)
    op_score.set_weight(rosetta.core.scoring.rna_sugar_close, 9.0)
    op_score.set_weight(rosetta.core.scoring.fa_intra_rep, 9)
    op_score.set_weight(rosetta.core.scoring.rna_base_pair, 9)
    op_score.set_weight(rosetta.core.scoring.rna_base_stack, 9)

    cutoff_dist = args.dcut
    cutoff_angle = 0.75
    cutoff_dihedral = 0.85
    cutoff_cont = 0.6

    cstpath = args.tmpdir + f'/cstfile_dist.txt'
    cstpath_cont = args.tmpdir + f'/cstfile_cont.txt'

    allcst = read_cst(cstpath)
    allcst_cont = read_cst(cstpath_cont)

    # cst_op=fetch_cst_atomset(allcst,atomset)
    cst_op = allcst + allcst_cont
    pcut = {
        'AtomPair': cutoff_dist,
        'Dihedral': cutoff_dihedral,
    }

    nres = len(read_fasta(args.FASTA))
    """  \mode=1\\"""
    # using all cst
    sep1 = 1
    sep2 = 10000

    minstd = 0.01
    std_cut = minstd  # +0.01*pcut['AtomPair']
    cst_all = fetch_cst(cst_op, sep1, sep2, pcut, std_cut)

    cstpath_all = args.tmpdir + f'/cstfile.txt'
    F = open(cstpath_all, "w")
    for a in cst_all:
        F.write(a)
        F.write("\n")
    F.close()

    executor = concurrent.futures.ProcessPoolExecutor(args.CPU)
    futures = [executor.submit(fold_single, cstpath_all, args) for _ in range(args.nmodels)]
    results = concurrent.futures.wait(futures)
    poses = list(results[0])

    min_energy = np.inf
    for pose in poses:
        pose = pose.result()
        energy = op_score(pose)
        if energy < min_energy:
            best_pose = pose
            min_energy = energy

    name = args.OUT
    best_pose.dump_pdb(name)
    print('\ndone')


def fold_single(cst_file, args):
    mmap = MoveMap()
    mmap.set_bb(True)  ##Whether the frame dihedral angle changes
    mmap.set_chi(True)  ##Whether the dihedral angle of the side chain changes
    mmap.set_jump(True)  ##Relative movement between polypeptide chains

    min_mover = rosetta.protocols.minimization_packing.MinMover(mmap, op_score, 'lbfgs_armijo_nonmonotone', 0.0001,
                                                                True)
    min_mover.max_iter(10000)

    repeat_mover = RepeatMover(min_mover, 5)

    mmap_ = MoveMap()
    mmap_.set_bb(True)  ##Whether the frame dihedral angle changes
    mmap_.set_chi(True)  ##Whether the dihedral angle of the side chain changes
    mmap_.set_jump(True)  ##Relative movement between polypeptide chains

    #################this part desinged for all-atom relax##################
    #  relax = rosetta.protocols.relax.FastRelax()
    #  relax.set_scorefxn(op_score)
    #  relax.max_iter(1000)
    #  relax.dualspace(True)
    #  relax.set_movemap(mmap_)
    ################## this part desiged for minimizer######################
    # rna_min_options = core.import_pose.options.RNA_MinimizerOptions()
    # rna_min_options.set_max_iter(1000)
    # rna_minimizer = protocols.rna.denovo.movers.RNA_Minimizer(rna_min_options)
    # rna_minimizer.set_score_function(op_score)

    assembler = core.import_pose.RNA_HelixAssembler()
    initpose = assembler.build_init_pose(args.seq, '')  # helix pose
    pose = basic_folding(initpose)
    pose.remove_constraints()

    run_min(cst_file, pose, repeat_mover, min_mover)
    run_refine(pose)

    return pose


def randTrial(your_pose):
    randNum = random.randint(2, your_pose.total_residue())

    curralpha = your_pose.alpha(randNum)
    currbeta = your_pose.beta(randNum)
    currgamma = your_pose.gamma(randNum)
    currdelta = your_pose.delta(randNum)
    currepsilon = your_pose.epsilon(randNum)
    currzeta = your_pose.zeta(randNum)
    currchi = your_pose.chi(randNum)

    newalpha = random.gauss(curralpha, 25)
    newbeta = random.gauss(currbeta, 25)
    newgamma = random.gauss(currgamma, 25)
    newdelta = random.gauss(currdelta, 25)
    newepsilon = random.gauss(currepsilon, 25)
    newzeta = random.gauss(currzeta, 25)
    newchi = random.gauss(currchi, 25)

    your_pose.set_alpha(randNum, newalpha)
    your_pose.set_beta(randNum, newbeta)
    your_pose.set_gamma(randNum, newgamma)
    your_pose.set_delta(randNum, newdelta)
    your_pose.set_epsilon(randNum, newepsilon)
    your_pose.set_zeta(randNum, newzeta)
    your_pose.set_chi(randNum, newchi)

    return your_pose


def decision(before_pose, after_pose):
    E = score(after_pose) - score(before_pose)
    if E < 0:
        return after_pose
    elif random.uniform(0, 1) >= math.exp(-E / 1):
        return before_pose
    else:
        return after_pose


def basic_folding(your_pose):
    lowest_pose = Pose()  # Create an empty pose for tracking the lowest energy pose.
    for i in range(120):
        if i == 0:
            lowest_pose.assign(your_pose)

        before_pose = Pose()
        before_pose.assign(your_pose)  # keep track of pose before random move

        after_pose = Pose()
        after_pose.assign(randTrial(your_pose))  # do rand move and store the pose

        your_pose.assign(decision(before_pose, after_pose))  # keep the new pose or old pose

        if score(your_pose) < score(lowest_pose):  # updating lowest pose
            lowest_pose.assign(your_pose)

    return lowest_pose


def score(your_pose):
    sf = rna_lowres_sf(your_pose)
    return sf


def read_fasta(file):
    fasta = "";
    with open(file, "r") as f:
        for line in f:
            if (line[0] == ">"):
                continue
            else:
                line = line.rstrip()
                fasta = fasta + line;
    return fasta


def read_cst(file):
    array = []
    with open(file, "r") as f:
        for line in f:
            line = line.rstrip()
            array.append(line)
    return array


def add_cst(pose, cstfile):
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(cstfile)
    constraints.add_constraints(True)
    constraints.apply(pose)


def remove_clash(scorefxn, mover, pose):
    clash_score = float(scorefxn(pose))
    if (clash_score > 10):
        for nm in range(0, 10):
            mover.apply(pose)
            clash_score = float(scorefxn(pose))
            if (clash_score < 10): break


def fetch_cst(cst, sep1, sep2, cut, std_cut=None):
    array = []
    for line in cst:
        # print(line)
        line = line.rstrip()
        b = line.split()
        cst_name = b[0]
        pcut = cut[cst_name]

        if std_cut is not None:
            if not line.endswith('#cont'):
                p_std = float(line.split('#')[1].split()[0])
            else:
                p_std = 10000
        if line.endswith('#cont'):
            prob = 1
        else:
            prob = float(b[-1])

        i = int(b[2])
        j = int(b[4])

        sep = abs(j - i)
        if (sep < sep1 or sep >= sep2): continue
        if std_cut is not None and p_std < std_cut: continue
        if (prob >= pcut):
            array.append(line)
    return array


# def run_min(array, pose, mover, tmpname):
# add_cst(pose, array, tmpname)
# mover.apply(pose)

def run_min(cstfile, pose, mover1, mover2=None):
    add_cst(pose, cstfile)
    mover1.apply(pose)
    if mover2 is not None:
        mover2.apply(pose)


def run_refine(pose):
    idealize = rosetta.protocols.idealize.IdealizeMover()
    poslist = rosetta.utility.vector1_unsigned_long()

    scorefxn = create_score_function('empty')
    scorefxn.set_weight(rosetta.core.scoring.cart_bonded, 1.0)
    scorefxn.score(pose)

    emap = pose.energies()
    # print("idealize...")
    for res in range(1, len(pose.residues) + 1):
        cart = emap.residue_total_energy(res)
        if cart > 50:
            poslist.append(res)
            # print("idealize %d %8.3f" % (res, cart))

    if len(poslist) > 0:
        idealize.set_pos_list(poslist)
    try:
        idealize.apply(pose)

        # cart-minimize
        scorefxn_min = create_score_function('ref2015_cart')

        mmap = MoveMap()
        mmap.set_bb(True)
        mmap.set_chi(True)
        mmap.set_jump(True)
        mmap.set_chi(False)

        min_mover = rosetta.protocols.minimization_packing.MinMover(mmap, scorefxn_min, 'lbfgs_armijo_nonmonotone',
                                                                    0.00001, True)
        # min_mover = rosetta.protocols.minimization_packing.MinMover(mmap, op_score, 'lbfgs_armijo_nonmonotone', 0.00001, True)
        min_mover.max_iter(200)
        min_mover.cartesian(True)
        # print("minimize...")
        min_mover.apply(pose)

    except:
        print('!!! idealization failed !!!')
