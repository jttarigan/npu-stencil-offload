device	workload	path	mode	t_sub K=1..8 (ms)	a fixed	b per step	R2	per-step gain K=8 meas	pred	ceiling (a+b)/b
SM-T975	grayscott_s8_ec	litert-cpu	b2b	7.817 / 15.946 / 34.458 / 71.084	-1.759	9.089	1.000	0.88x	0.83x	0.81x
SM-T975	grayscott_s8_ec	litert-cpu	paced	20.459 / 33.721 / 48.062 / 78.065	15.271	7.948	0.991	2.10x	2.36x	2.92x
SM-T975	grayscott_s8_ec	litert-gpu	b2b	3.498 / 5.259 / 14.253 / 18.653	2.066	2.227	0.908	1.50x	1.73x	1.93x
SM-T975	grayscott_s8_ec	litert-gpu	paced	6.531 / 12.973 / 23.633 / 34.804	4.724	3.936	0.964	1.50x	1.91x	2.20x
SM-T975	grayscott_s8_ec	nnapi	b2b	61.680 / 101.013 / 166.328 / 312.161	27.008	35.543	1.000	1.58x	1.61x	1.76x
SM-T975	grayscott_s8_ec	nnapi	paced	60.488 / 99.743 / 167.536 / 313.395	25.562	35.928	1.000	1.54x	1.57x	1.71x
SM-T975	heat_s8_ec	litert-cpu	b2b	2.159 / 3.955 / 7.981 / 16.201	0.010	2.017	1.000	1.07x	1.00x	1.01x
SM-T975	heat_s8_ec	litert-cpu	paced	8.981 / 15.318 / 22.988 / 36.880	6.566	3.860	0.990	1.95x	2.23x	2.70x
SM-T975	heat_s8_ec	litert-gpu	b2b	1.632 / 1.968 / 3.180 / 5.999	0.800	0.639	0.993	2.18x	1.95x	2.25x
SM-T975	heat_s8_ec	litert-gpu	paced	2.830 / 4.464 / 14.247 / 16.662	1.884	2.044	0.837	1.36x	1.72x	1.92x
SM-T975	heat_s8_ec	nnapi	b2b	27.359 / 34.407 / 63.425 / 124.797	8.915	14.289	0.994	1.75x	1.51x	1.62x
SM-T975	heat_s8_ec	nnapi	paced	30.634 / 47.286 / 81.598 / 145.523	14.711	16.413	1.000	1.68x	1.71x	1.90x
SM-T975	ink_i32_ec_fs	litert-cpu	b2b	12.021 / 25.767 / 52.539 / 105.420	-1.022	13.322	1.000	0.91x	0.93x	0.92x
SM-T975	ink_i32_ec_fs	litert-cpu	paced	22.856 / 33.304 / 55.616 / 104.290	10.190	11.687	0.999	1.75x	1.69x	1.87x
SM-T975	ink_i32_ec_fs	nnapi	b2b	95.779 / 158.862 / 285.049 / 584.804	18.261	70.097	0.998	1.31x	1.22x	1.26x
SM-T975	ink_i32_ec_fs	nnapi	paced	97.846 / 154.850 / 287.197 / 586.159	17.655	70.362	0.998	1.34x	1.21x	1.25x
