#======================================================================================================
# !/usr/bin/env python
# title          : LocalW2W_FWC_Run.py
# description    : Semiconductor Fab Wide Control using FDC, VM, R2R, L2L
# author         : Youngil Jung
# date           : 2019-06-17
# version        : v0.8
# usage          : python LocalW2W_FWC_Run.py
# notes          : Reference Paper "Virtual metrology and feedback control for semiconductor manufacturing"
# python_version : v3.5.3
#======================================================================================================
import numpy as np
from simulator.VM_Process1_DynamicSampling_노이즈시뮬레이터 import VM_Process1_DynamicSampling_노이즈시뮬레이터
from simulator.FDC_Graph import FDC_Graph
# from pandas import DataFrame, Series
# import pandas as pd
import os
from matplotlib import pyplot as plt

os.chdir("D:/10. 대학원/04. Source/09. VM_Source/10. DynamicSampling/")

A_p1 = np.array([[0.5, -0.2], [0.25, 0.15]])    #recipe gain matrix
d_p1 = np.array([[0.1, 0], [0.05, 0]])  #drift matrix
C_p1 = np.transpose(np.array([[0, 0.5, 0.05, 0, 0.15, 0], [0.085, 0, 0.025, 0.2, 0, 0]])) # FDC variable matrix
SEED = 111999999  #3

M = 10
S1 = 20
dM = 5
S2 = 60
RUNS_CNT = S1 + S2
N = M * S1 + dM * S2
dStart = S1 + 1  #25 * 10 + 30 * 5
Z_DoE = 12
v_PLS = 0.6
Nz_RUN = 15

def main():
    fdh_graph = FDC_Graph()
    fwc_p1_vm = VM_Process1_DynamicSampling_노이즈시뮬레이터(A_p1, d_p1, C_p1, dM, dStart, SEED)
    fwc_p1_vm.DoE_Run(lamda_PLS=v_PLS, Z=Z_DoE, M=M)  #DoE Run
    VM_Output, ACT_Output, ez_run, y_act, y_prd = fwc_p1_vm.VM_Run(lamda_PLS=v_PLS, Z=RUNS_CNT, M=M)

    np.savetxt("output/VM_Output.csv", VM_Output, delimiter=",", fmt="%.4f")
    np.savetxt("output/ACT_Output.csv", ACT_Output, delimiter=",", fmt="%.4f")
    np.savetxt("output/ez_run.csv", ez_run, delimiter=",", fmt="%.4f")
    np.savetxt("output/y_act.csv", y_act, delimiter=",", fmt="%.4f")
    np.savetxt("output/y_prd.csv", y_prd, delimiter=",", fmt="%.4f")

    fdh_graph.plt_show1(N, y_act[:, 0:1], y_prd[:, 0:1])
    fdh_graph.plt_show2(RUNS_CNT, ez_run[:, 0:1], ez_run[:, 1:2], Noise=True)

    ez_run_out = []
    runM = M

    ez_run_out.append(np.array([0, 0]))
    for z in np.arange(1, RUNS_CNT + 1):
        if z == dStart:
            runM = dM
        for k in np.arange(z * runM, (z + 1) * runM):
            ez_run_out.append(ez_run[z])
    ez_run_out = np.array(ez_run_out)

    np.savetxt("output/ez_run2.csv", ez_run_out, delimiter=",", fmt="%.4f")
    fdh_graph.plt_show5(ez_run_out, N, M, dM, S1, Noise=True)

    p1_q1_mape_Queue = []

    #metrology 마다 보여주는 MAPE 값이 의미가 없다. ==> 로직 고쳐야 함.. 40 runs가 아니라 60 Runs, 80 Runs 변경 계산 필요
    for z in np.arange(Nz_RUN, 40, 1):
        mape = fdh_graph.mean_absolute_percentage_error(z + 1, y_act[((z + 1) * M) - 1][0], y_prd[((z + 1) * M) - 1][0])
        p1_q1_mape_Queue.append(mape)

    print('Process-1 q1 Every Metrology MAPE After 15 Lot : {0:.2f}%'.format(np.mean(p1_q1_mape_Queue)))
    p1_q1_mape_Queue = []

    for i in np.arange(Nz_RUN * M, N, 1):
        mape = fdh_graph.mean_absolute_percentage_error(i + 1, y_act[i][0], y_prd[i][0])
        p1_q1_mape_Queue.append(mape)

    print('Process-1 q1 All MAPE After 15 Lot : {0:.2f}%'.format(np.mean(p1_q1_mape_Queue)))

if __name__ == "__main__":
    main()
