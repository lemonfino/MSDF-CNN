Kstep = 3;
alpha = 0.98;

Nets = {'Sim_mat_drug_ATC','Sim_mat_drug_enzymes','Sim_mat_drug_protein','Sim_mat_drug_sideeffects','Sim_mat_drug_smiles','Sim_mat_protein_pathways','Sim_mat_protein_protein'};

for i = 1 : length(Nets)
	tic
	inputID = char(strcat('F:/lj-FR_CNN/12-simNet/', Nets(i), '.txt'));
	data1 = load(inputID);
	data2 = RandSurf(data1, Kstep, alpha);
    PPMI = GetPPMIMatrix(data2);
	outputID = char(strcat('F:/lj-FR_CNN/feature/PPMI_', Nets(i), '.txt'));
	dlmwrite(outputID, PPMI, '\t');
	toc
end

PPMI1 = load('F:/lj-FR_CNN/feature/PPMI_Sim_mat_drug_ATC.txt');
PPMI2 = load('F:/lj-FR_CNN/feature/PPMI_Sim_mat_drug_enzymes.txt');
PPMI3 = load('F:/lj-FR_CNN/feature/PPMI_Sim_mat_drug_protein.txt');
PPMI4 = load('F:/lj-FR_CNN/feature/PPMI_Sim_mat_drug_sideeffects.txt');
PPMI5 = load('F:/lj-FR_CNN/feature/PPMI_Sim_mat_drug_smiles.txt');
PPMI6 = load('F:/lj-FR_CNN/feature/PPMI_Sim_mat_protein_pathways.txt');
PPMI7 = load('F:/lj-FR_CNN/feature/PPMI_Sim_mat_protein_protein.txt');

drug_feature = [PPMI1,PPMI2,PPMI3,PPMI4,PPMI5];
protein_feature = [PPMI6,PPMI7];

dlmwrite(['F:/lj-FR_CNN/feature/drug_feature.txt'], drug_feature, '\t');
dlmwrite(['F:/lj-FR_CNN/feature/protein_feature.txt'], protein_feature, '\t');

