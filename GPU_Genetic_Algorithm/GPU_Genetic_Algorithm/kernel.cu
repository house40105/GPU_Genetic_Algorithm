#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#define M_PI 3.14159265359
#define POPULATION_CNT 100// 母群數量
#define CROSSOVER_RATE 0.8// 交配率
#define MUTATION_RATE 0.001// 突變率

typedef struct coordinate{
	int no;//胺基酸內原子編號
	char name[3];//元素名
	char Aname[4];//胺基酸名稱
	double x,y,z;
	int group;// 0非連接點 1連接點
} Coordinate;
typedef struct tag_degree_t{
	double theta;
	double Phi;
	double Psi;
	double W;
}degree_t;
typedef struct tag_parent_t{
	//degree_t degree_t;
	degree_t *degree_t;
	double fitness;//適應值
}parent_t;
typedef struct protein_distance{
	//int A1_No, A2_No;//胺基酸序列上編號
	char A1_atom[3], A2_atom[3];//原子名稱
	//char A1_name[4], A2_name[4];//胺基酸名稱
	//int A1_atom_No, A2_atom_No;//胺基酸原子編號
	double distance;//距離
}P_distance;

parent_t population[POPULATION_CNT];	//母體數量
parent_t pool[POPULATION_CNT];			//交配池
parent_t best_gene[5];					//從以前到現在最好的基因


//-----------------------序列座標初始化---------------------------
void LoadData(Coordinate data1[26][30]);//胺基酸資料讀取
int CheckAmino(int *input, int c);
void Connection(int *Index,Coordinate Data[26][30], int G_LENGTH, Coordinate **PredictChain);  //連接胺基酸 初始化座標
void OutPut(Coordinate **PredictChain,int c,char fName[50]);
//----------------------Genetic Algorithm-------------------------
double fRand(double fMin, double fMax);//0~360度角隨機小數亂數
__device__ double Electrostatic(char Q1[],char Q2[],double R);//(kcal/mol)
__device__ double VDW(char A1[],char A2[],double R);//(kcal/mol)
__global__ void cal_fitness(parent_t *dev_population,P_distance *dev_A_distance,int Atom_sum,int Space);//適應函式
void sortfunction();
void insert_bestgene();
void initialize(int G_LENGTH);//初始化Chromosome
void reproduction();//複製, 輪盤式選擇(分配式)
void crossover(int G_LENGTH);//交配
void mutation(int G_LENGTH);//突變
//---------------------Structure Transform------------------------
void RotationMatrix(double angle, double a,double b, double c, double u, double v, double w,
					double inputMatrix[4][1],double outputMatrix[4][1]);//旋轉函式
void Transform(Coordinate ***chromosome, int G_LENGTH);//結構轉換
void Peptidebond(double x[], double y[], double z[], double F[]);//肽鍵鍵結
void Binding_Correction(Coordinate **PredictChain, int location);//鍵結修正
__global__ void Distance(P_distance *dev_A_distance, int A_sum, int S_sum, Coordinate *dev_chromosome, int GENETIC_LENGTH);//原子距離
void Find_bestgene(Coordinate ***chromosome,Coordinate **BestChain, int G_LENGTH);//搜尋最佳結構

int main(void)
{
	char username[50];			//存檔名
	Coordinate Data[26][30];	//胺基酸資料
	char *inputWord, temp;
	int *inputIndex;
	Coordinate **PredictChain;	//串接後的序列
	int GENETIC_LENGTH=0;		//基因長度
	int ITERA_CNT=50;			//世代次數
	Coordinate ***chromosome;	//染色體
	FILE *fPtr;					//讀入序列檔
	int Atom_sum=0, count=0, Space_sum=0;	//原子總數, 計數
	P_distance **Atom_distance;//原子間彼此的距離
	Coordinate **BestChain;		//最佳序列

	Coordinate *dev_chromosome=0;	//CUDA染色體
	P_distance *dev_Atom_distance=0;//CUDA原子距離
	parent_t *dev_population=0;	//CUDA母體數量
	cudaError_t cudaStatus;
	time_t start_time, finish_time;
	
	///設定GPU
	cudaStatus=cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
        //printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		//system("pause");
        return 1;
    }

	start_time=clock();

	

	LoadData(Data);
	GENETIC_LENGTH=0;
	fPtr = fopen("test.txt","r");
	//--------------------取檔名
	fscanf(fPtr,"%s",username);
	fscanf(fPtr,"%c",&temp);//取 \n
	//--------------------
	while(!feof(fPtr)){  //計算基因長度
		fscanf(fPtr,"%c",&temp);
		GENETIC_LENGTH++;
	}
	GENETIC_LENGTH--;

	inputWord = (char *)malloc(sizeof(char)*GENETIC_LENGTH);
	inputIndex = (int *)malloc(sizeof(int)*GENETIC_LENGTH);
	PredictChain = (Coordinate **)malloc(sizeof(Coordinate *)*GENETIC_LENGTH);
	BestChain = (Coordinate **)malloc(sizeof(Coordinate *)*GENETIC_LENGTH);
	for(int i=0; i<GENETIC_LENGTH; i++)
	{
		PredictChain[i] = (Coordinate *)malloc(sizeof(Coordinate)*30);
		BestChain[i] = (Coordinate *)malloc(sizeof(Coordinate)*30);
	}
	rewind(fPtr);  //檔案重頭
	//--------------------取檔名
	fscanf(fPtr,"%s",username);
	fscanf(fPtr,"%c",&temp);//取 \n
	//--------------------
	for(int i=0; i<GENETIC_LENGTH; i++)//讀檔
	{
		fscanf(fPtr,"%c",&inputWord[i]);
	}
	for(int i=0; i<GENETIC_LENGTH; i++)//轉大寫
	{
		inputWord[i]=toupper(inputWord[i]);
	}
	for(int i=0; i<GENETIC_LENGTH; i++)
	{
		inputIndex[i]=inputWord[i]-'A';
	}
	if(CheckAmino(inputIndex,GENETIC_LENGTH)!=0){//找不到
		Connection(inputIndex, Data, GENETIC_LENGTH, PredictChain);
		//OutPut(PredictChain, GENETIC_LENGTH);//結構輸出

		srand(time(NULL));
		initialize(GENETIC_LENGTH);//初始化

		chromosome = (Coordinate ***)malloc(sizeof(Coordinate **)*POPULATION_CNT);//配置母群體空間
		for(int i=0; i<POPULATION_CNT; i++){
			chromosome[i] = (Coordinate **)malloc(sizeof(Coordinate *)*GENETIC_LENGTH);
			for(int j=0; j<GENETIC_LENGTH; j++){
				chromosome[i][j] = (Coordinate *)malloc(sizeof(Coordinate)*30);
			}
		}

		for(int i=0; i<POPULATION_CNT; i++){//載入蛋白質
			for(int j=0; j<GENETIC_LENGTH; j++)
				for(int k=0;k<30;k++){
					memcpy(&chromosome[i][j][k], &PredictChain[j][k],sizeof(Coordinate));
				}
		}

		Transform(chromosome, GENETIC_LENGTH);//結構轉換

		Atom_sum=GENETIC_LENGTH*30;//原子數計
		for(int i=Atom_sum; i>0; i--){
			Space_sum += i;
		}
		//==========================================cuda=====================================================

		
		cudaStatus=cudaMalloc((void **) &dev_chromosome,POPULATION_CNT*GENETIC_LENGTH*30*sizeof(Coordinate));
		
		if (cudaStatus != cudaSuccess) {
			//printf("1cudaMalloc failed!\n");
			//system("pause");
			return 1;
		}
		for(int i=0; i<POPULATION_CNT; i++){
			for(int j=0; j<GENETIC_LENGTH; j++){
				for(int k=0; k<30; k++){
					//printf("in host %d %s \n", i*GENETIC_LENGTH*30+j*30+k, chromosome[i][j][k].name);
					cudaMemcpy(&dev_chromosome[i*GENETIC_LENGTH*30+j*30+k],&chromosome[i][j][k],sizeof(Coordinate),cudaMemcpyHostToDevice);
				}
			}
		}
		//==========================================cuda=====================================================^
		Atom_distance = (P_distance **)malloc(sizeof(P_distance *)*POPULATION_CNT);//原子距離空間配置
		for(int i=0; i<POPULATION_CNT; i++){
			Atom_distance[i] = (P_distance *)malloc(sizeof(P_distance )*Space_sum);
		}
		//==========================================cuda=====================================================
		
		cudaStatus=cudaMalloc((void **) &dev_Atom_distance, (POPULATION_CNT*Space_sum)*sizeof(P_distance));
		if (cudaStatus != cudaSuccess) {
			//printf("2cudaMalloc failed!");
			//system("pause");
			return 1;
		}

		//==========================================cuda=====================================================^
		Distance<<<1, POPULATION_CNT>>>(dev_Atom_distance, Atom_sum, Space_sum, dev_chromosome, GENETIC_LENGTH);//距離計算
		cudaThreadSynchronize();

		//printf("OK\n");
		//==========================================cuda=====================================================
		cudaStatus=cudaMalloc((void **) &dev_population,sizeof(parent_t)*POPULATION_CNT);
		if (cudaStatus != cudaSuccess) {
			//printf("3cudaMalloc failed!");
			//system("pause");
			return 1;
		}
		//==========================================cuda=====================================================^
		cal_fitness<<<1, POPULATION_CNT>>>(dev_population,dev_Atom_distance,Atom_sum,Space_sum);//給分數
		cudaThreadSynchronize();

		for(int i=0; i<POPULATION_CNT; i++){
			cudaMemcpy(&population[i].fitness,&dev_population[i].fitness,sizeof(double),cudaMemcpyDeviceToHost);
		}

		//printf("popu %lf %lf\n", population[0].degree_t->Phi, population[0].fitness);
		Find_bestgene(chromosome, BestChain,GENETIC_LENGTH);//搜尋最佳結構
		sortfunction(); 
		insert_bestgene();

		for(int k=0;k<ITERA_CNT;k++)
		{
			if(ITERA_CNT<1)
				break;
			reproduction();                   // 複製,輪盤式選擇(分配式) 
			crossover(GENETIC_LENGTH);        // 交配
			mutation(GENETIC_LENGTH);         // 突變

			for(int i=0; i<POPULATION_CNT; i++){//載入蛋白質
				for(int j=0; j<GENETIC_LENGTH; j++)
					memcpy(chromosome[i][j], PredictChain[j],30*sizeof(Coordinate));
			}

			Transform(chromosome, GENETIC_LENGTH);

			for(int i=0; i<POPULATION_CNT; i++){
				for(int j=0; j<GENETIC_LENGTH; j++){
				//for(int k=0; k<30; k++)
					cudaMemcpy(&dev_chromosome[i*GENETIC_LENGTH*30+j*30],chromosome[i][j],sizeof(Coordinate)*30,cudaMemcpyHostToDevice);
				}
			}


			Distance<<<1, POPULATION_CNT>>>(dev_Atom_distance, Atom_sum, Space_sum, dev_chromosome, GENETIC_LENGTH);//距離計算
			cudaThreadSynchronize();

			cal_fitness<<<1, POPULATION_CNT>>>(dev_population,dev_Atom_distance,Atom_sum,Space_sum);//給分數
			cudaThreadSynchronize();

			for(int i=0; i<POPULATION_CNT; i++){
				cudaMemcpy(&population[i].fitness,&dev_population[i].fitness,sizeof(double),cudaMemcpyDeviceToHost);
			}
			Find_bestgene(chromosome, BestChain, GENETIC_LENGTH);//搜尋最佳結構
			sortfunction(); 
			insert_bestgene();
		}

		OutPut(BestChain, GENETIC_LENGTH,username);//結構輸出
	
		//free memory
		free(population->degree_t);
		for(int i=0; i<POPULATION_CNT; i++){
			free(chromosome[i]);
		}
		free(chromosome);
		for(int i=0; i<POPULATION_CNT; i++){
			free(Atom_distance[i]);
		}
		free(Atom_distance);
	}
	free(inputWord);
	free(inputIndex);
	for(int i=0; i<GENETIC_LENGTH; i++){
		free(PredictChain[i]);
	}
	free(PredictChain);
	free(BestChain);

	fclose(fPtr);
	finish_time=clock();
	printf("\n%lf\n", (finish_time-start_time)/(double)(CLOCKS_PER_SEC));

	//重設 GPU
	cudaDeviceReset();
	system("pause");
	return 0;
}


//-- 改寫 strcpy 與 strcmp
__device__ int dev_strcmp (const char * src, const char * dst)
{
        int ret = 0 ;

        while( ! (ret = *(unsigned char *)src - *(unsigned char *)dst) && *dst)
                ++src, ++dst;

        if ( ret < 0 )
                ret = -1 ;
        else if ( ret > 0 )
                ret = 1 ;

        return( ret );
}

__device__ void dev_strcpy(char *dest, const char *src, const int x)
{
	for(int i=0; i<x; i++){
		dest[i]=src[i];
	}
}


//----------------------序列座標初始化----------------------------
void LoadData(Coordinate data1[26][30])
{
	FILE *fPtr;
	char temp[10];

	Coordinate data[20][30];

	fPtr=fopen("data.txt","r");
	for(int i=0; i<26; i++)
	{
		fscanf(fPtr,"%s",temp);
		for(int j=0; j<30; j++)
		{
			fscanf(fPtr,"%s",temp);
			if(temp[0] =='*')
				break;
			fscanf(fPtr,"%d %s %s %s %lf %lf %lf %d",&data1[i][j].no,data1[i][j].name,data1[i][j].Aname,temp,&data1[i][j].x,&data1[i][j].y,&data1[i][j].z, &data1[i][j].group);
		}
	}
	fclose(fPtr);
}
int CheckAmino(int *input, int c)
{//胺基酸序列檢查副函式
	for(int i=0; i<c; i++)
	{
		if(input[i]==1 || input[i]==9 || input[i]==14 || input[i]==20 || input[i]==23 || input[i]==25)
		{	
			return 0;//找不到
			break;				
		}
	}
	return 1; //OK 沒問題 
}
void Connection(int *Index,Coordinate Data[26][30], int G_LENGTH, Coordinate **PredictChain)//連接胺基酸 初始化座標
{//胺基酸連接副函式
	//Coordinate PredictChain[20][30];
	double F_x,F_y,F_z;
	double inputMatrix[4][1] = {0.0, 0.0, 0.0, 0.0};//旋轉點
	double outputMatrix[4][1] = {0.0, 0.0, 0.0, 0.0};//旋轉後點
	double a; double b; double c;//旋轉軸通過點(a, b, c)
	double angle=180; double u; double v; double w;//角度、旋轉軸(u, v, w)
	double x[3], y[3], z[3], F[3];//C座標x、CA座標y、O座標z、鍵結座標F

	for(int i=0; i<G_LENGTH; i++)// i 胺基酸個數
	{
		int j=0;//第一顆原子的座標 歸零
		if(strcmp(Data[ Index[i] ][j].Aname, "PRO") == 0){
			F_x = Data[ Index[i] ][10].x;
			F_y = Data[ Index[i] ][10].y;
			F_z = Data[ Index[i] ][10].z;
		}
		else{
			F_x = Data[ Index[i] ][j].x;
			F_y = Data[ Index[i] ][j].y;
			F_z = Data[ Index[i] ][j].z;
		}
		while(Data[ Index[i] ][j].no > 0)
		{
			PredictChain[i][j].no = Data[ Index[i] ][j].no;
			PredictChain[i][j].group = Data[ Index[i] ][j].group;// int group;// 0非連接點 1連接點
			strcpy(PredictChain[i][j].name,Data[ Index[i] ][j].name);
			strcpy(PredictChain[i][j].Aname,Data[ Index[i] ][j].Aname);
			PredictChain[i][j].x = Data[ Index[i] ][j].x - F_x;//減掉 第一顆原子的座標
			PredictChain[i][j].y = Data[ Index[i] ][j].y - F_y;
			PredictChain[i][j].z = Data[ Index[i] ][j].z - F_z;
			j++;
			
		}
	}
	//串接
	
	F_x = PredictChain[0][0].x;
	F_y = PredictChain[0][0].y;
	F_z = PredictChain[0][0].z;//第一顆原子的坐標
	
	for(int i=0; i<G_LENGTH; i++)// i 胺基酸個數
	{
		int j=0;
		int p=0;//串接的位置
		if( strcmp(PredictChain[i][0].Aname, "PRO") == 0 ){//找旋轉軸
			a = PredictChain[i][10].x;
			b = PredictChain[i][10].y;
			c = PredictChain[i][10].z;
			u = PredictChain[i][11].x - PredictChain[i][10].x;
			v = PredictChain[i][11].y - PredictChain[i][10].y;
			w = PredictChain[i][11].z - PredictChain[i][10].z;
		}
		else {
			a = PredictChain[i][0].x;
			b = PredictChain[i][0].y;
			c = PredictChain[i][0].z;
			u = PredictChain[i][2].x - PredictChain[i][0].x;
			v = PredictChain[i][2].y - PredictChain[i][0].y;
			w = PredictChain[i][2].z - PredictChain[i][0].z;
		}
		while(Data[ Index[i] ][j].no > 0)
		{
			if(i%2==1){//----------------------------------------------------------------------------
				inputMatrix[0][0]=PredictChain[i][j].x;
				inputMatrix[1][0]=PredictChain[i][j].y;
				inputMatrix[2][0]=PredictChain[i][j].z;
				inputMatrix[3][0]=1.0;
				RotationMatrix(angle, a, b, c, u, v, w,inputMatrix,outputMatrix);//旋轉
				PredictChain[i][j].x = outputMatrix[0][0];
				PredictChain[i][j].y = outputMatrix[1][0];
				PredictChain[i][j].z = outputMatrix[2][0];
			}
			PredictChain[i][j].x = PredictChain[i][j].x + F_x;//+上一顆原子的座標
			PredictChain[i][j].y = PredictChain[i][j].y + F_y;
			PredictChain[i][j].z = PredictChain[i][j].z + F_z;
			if(PredictChain[i][j].group == 2 && strcmp(PredictChain[i][j].name,"C")==0)//mark 連接處(C座標)
				p=j;
			j++;
		}

		if(i != 0)
			Binding_Correction(PredictChain, i);

		for(int w=0; PredictChain[i][w].no>0; w++){
			if(PredictChain[i][w].group == 2 && strcmp(PredictChain[i][w].name,"C")==0)//mark 連接處(C座標)
			{
				x[0] = PredictChain[i][w].x;
				x[1] = PredictChain[i][w].y;
				x[2] = PredictChain[i][w].z;
			}
			else if(PredictChain[i][w].group == 1 && strcmp(PredictChain[i][w].name,"CA")==0)//mark 連接處(CA座標)
			{
				y[0] = PredictChain[i][w].x;
				y[1] = PredictChain[i][w].y;
				y[2] = PredictChain[i][w].z;
			}
			else if(PredictChain[i][w].group == 2 && strcmp(PredictChain[i][w].name,"O")==0)//mark 連接處(O座標)
			{
				z[0] = PredictChain[i][w].x;
				z[1] = PredictChain[i][w].y;
				z[2] = PredictChain[i][w].z;
			}
		}
		F[0] = 0;//座標初始化
		F[1] = 0;
		F[2] = 0;
		Peptidebond(x, y, z, F);//肽鍵鍵結點
		F_x = F[0];//C-N 接 肽鍵
		F_y = F[1];
		F_z = F[2];
		
	}
}
void OutPut(Coordinate **PredictChain,int c, char fName[50])
{//PDB輸出副函式
	FILE *fPtr;
	int num=1;
	fPtr=fopen(fName,"w");
	fprintf(fPtr,"HEADER    AMINO ACID\n");
	fprintf(fPtr,"AUTHOR    GENERATED BY GLACTONE\n");
	for(int i=0; i<c; i++)
	{
		int j=0;
		while(PredictChain[i][j].no > 0)
		{
			fprintf(fPtr,"ATOM   %4d  %-3s %3s A %3d   %9.3f %7.3f %7.3f  1.00  0.00\n",num,PredictChain[i][j].name,PredictChain[i][j].Aname,i+1,PredictChain[i][j].x,PredictChain[i][j].y,PredictChain[i][j].z);
			j++;
			num++;
		}	
	}
	fprintf(fPtr,"END\n");
	fclose(fPtr);
}

//----------------------Genetic Algorithm-------------------------
double fRand(double fMin, double fMax)//0~360度角隨機小數亂數
{
	double f = (double)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}
__device__ double Electrostatic(char Q1[],char Q2[],double R)//(kcal/mol)
{
	//At moderate temperatures.
	//Q1,Q2 are two charges.
	//R is distance.
	double q1=0,q2=0;

	if(dev_strcmp(Q1,"H")==0)
	{
		q1=0.31;
	}
	else if(dev_strcmp(Q1,"N")==0)
	{
		q1=-0.47;
	}
	else if(dev_strcmp(Q1,"O")==0)
	{
		q1=-0.51;
	}
	else if(dev_strcmp(Q1,"C")==0)
	{
		q1=0.51;
	}
	else if(dev_strcmp(Q1,"CA")==0)
	{
		q1=0.07;
	}
	else if(dev_strcmp(Q1,"S")==0)
	{
		q1=-0.23;
	}
	//
	if(dev_strcmp(Q2,"H")==0)
	{
		q2=0.31;
	}
	else if(dev_strcmp(Q2,"N")==0)
	{
		q2=-0.47;
	}
	else if(dev_strcmp(Q2,"O")==0)
	{
		q2=-0.51;
	}
	else if(dev_strcmp(Q2,"C")==0)
	{
		q2=0.51;
	}
	else if(dev_strcmp(Q2,"CA")==0)
	{
		q2=0.07;
	}
	else if(dev_strcmp(Q2,"S")==0)
	{
		q2=-0.23;
	}
	double Epsilon=40; 

	double E_elec=322.0637*((q1*q2)/((Epsilon*R)*R));
	return E_elec;
}
__device__ double VDW(char A1[],char A2[],double R)
{
	//V(Lennard-Jones) = Eps,i,j[(Rmin,i,j/ri,j)**12 - 2(Rmin,i,j/ri,j)**6]
	//epsilon: kcal/mole, Eps,i,j = sqrt(eps,i * eps,j)
	//Rmin/2: A, Rmin,i,j = Rmin/2,i + Rmin/2,j
	double ESP=0;
	double esp1=0,esp2=0;
	double Rmin=0;
	double Rmin1=0,Rmin2=0;

	if(dev_strcmp(A1,"H")==0)
	{
		esp1=-0.046000;
		Rmin1=0.224500;
	}
	else if(dev_strcmp(A1,"N")==0)
	{
		esp1=-0.200000;
		Rmin1=1.850000;
	}
	else if(dev_strcmp(A1,"O")==0)
	{
		esp1=-0.120000;
		Rmin1=1.700000;
	}
	else if(dev_strcmp(A1,"C")==0)
	{
		esp1=-0.110000;
		Rmin1=2.000000;
	}
	else if(dev_strcmp(A1,"CA")==0)
	{
		esp1=-0.070000;
		Rmin1=1.992400;
	}
	else if(dev_strcmp(A1,"S")==0)
	{
		esp1=-0.450000;
		Rmin1=2.000000;
	}
	//
	if(dev_strcmp(A2,"H")==0)
	{
		esp2=-0.046000;
		Rmin2=0.224500;
	}
	else if(dev_strcmp(A2,"N")==0)
	{
		esp2=-0.200000;
		Rmin2=1.850000;
	}
	else if(dev_strcmp(A2,"O")==0)
	{
		esp2=-0.120000;
		Rmin2=1.700000;
	}
	else if(dev_strcmp(A2,"C")==0)
	{
		esp2=-0.110000;
		Rmin2=2.000000;
	}
	else if(dev_strcmp(A2,"CA")==0)
	{
		esp2=-0.070000;
		Rmin2=1.992400;
	}
	else if(dev_strcmp(A2,"S")==0)
	{
		esp2=-0.450000;
		Rmin2=2.000000;
	}

	ESP=sqrt(esp1*esp2);
	Rmin=Rmin1+Rmin2;
	double E_VDM=ESP*(pow((Rmin/R),12)-2*pow((Rmin/R),6));
	return E_VDM;
}
__global__ void cal_fitness(parent_t *dev_population,P_distance *dev_A_distance,int Atom_sum,int Space)//適應函式
{   
	double Total_Energy=0;
	double Energy=0;
	char S1[4], S2[4];
	int a=threadIdx.x;


	for(int j=0;j<Space;j++)
	{
		if(dev_A_distance[j].distance!=0)
		{
			Energy=Electrostatic(dev_A_distance[j].A1_atom,dev_A_distance[j].A1_atom,dev_A_distance[j].distance)+VDW(dev_A_distance[j].A1_atom,dev_A_distance[j].A1_atom,dev_A_distance[j].distance);
			Total_Energy+=Energy;
		}
	}
	
	dev_population[a].fitness=Total_Energy;
}
void sortfunction()
{
	parent_t x;
	int j = 0;

	for(int i=1;i<POPULATION_CNT;i++)
	{
		x = population[i];

		for(j=i-1;j>=0;j--)
		{
			if(population[j].fitness>x.fitness)
				population[j+1] = population[j];

			else
			{
				population[j+1] = x;
				break;
			}
		}

		if(j < 0)
			population[0] = x;
	}
}
void insert_bestgene()
{
	for(int i=0;i<5;i++)
	{
		best_gene[i]=population[i];
	}
}
void initialize(int G_LENGTH)//初始化
{
	int i, j;

	for(i=0;i<POPULATION_CNT;i++)
	{
		degree_t *degree=(degree_t*)malloc(sizeof(degree_t)*G_LENGTH);
		population[i].degree_t=degree;
		for(j=0;j<G_LENGTH;j++)
		{
			// 每個母體的基因都是隨機給0~360
			population[i].degree_t[j].theta = fRand(0,360);
			population[i].degree_t[j].Phi = fRand(0,360);
			population[i].degree_t[j].Psi = fRand(0,360);
			population[i].degree_t[j].W = fRand(0,360);
		}
	}
}
void reproduction()//複製
{   int p1=0;

	for(int i=0;i<POPULATION_CNT;i++)
	{
		p1 = rand() % 5;
		pool[i]=best_gene[p1];
	}
}
void crossover(int G_LENGTH)//交配
{
	int i, itera;
	int cnt=5;
	int pos=0;
	int p1, p2;
	double crossover_if;

	
	for(int x=0;x<5;x++)//菁英染色體留下
	{
		population[x]=best_gene[x];
	}

	for(itera = 5;itera < POPULATION_CNT;itera+=2)
	{
		// 隨機選二個個體
		p1 = rand() % POPULATION_CNT;     
		do{
			p2=rand()% POPULATION_CNT;
		}while(p2==p1);

		crossover_if = ((double)rand()/(double)RAND_MAX);// 決定是否交配
		if(crossover_if > CROSSOVER_RATE)
		{
			// 不交配, 將交配池中之個體丟回母體
			memcpy( (void *)&population[cnt++],(void *)&pool[p1],sizeof(parent_t));
			memcpy( (void *)&population[cnt++],(void *)&pool[p2],sizeof(parent_t));
		}
		else {
			// 單點交配,交配完後再丟回母體
			do{
				pos = (rand()%G_LENGTH);
			} while(pos==0);               
			// crossover
			for(i=0;i<pos;i++)
			{
				population[cnt].degree_t[i] = pool[p1].degree_t[i];
				population[cnt+1].degree_t[i] = pool[p2].degree_t[i];
			}
			for(i=pos;i<G_LENGTH;i++) 
			{
				population[cnt+1].degree_t[i] = pool[p1].degree_t[i];
				population[cnt].degree_t[i] = pool[p2].degree_t[i];
			}
			cnt+=2; //已複制完二條
			if(cnt>=POPULATION_CNT)
			{
				break;
			}
		}           
	}
}
void mutation(int G_LENGTH)//突變
{
	int i;
	int pos,pot;

	for(i=0;i<POPULATION_CNT;i++)
	{
		double mutation_if = ((double)rand()/(double)RAND_MAX);
		if(mutation_if <= MUTATION_RATE)
		{
			pos = (rand()%G_LENGTH);     // 突變位置
			do{            //隨機選擇Phi,Psi,W的突變點
				pot = (rand()%3); 
			}while((pos==0&&pot==0)||(pos==(G_LENGTH-1)&&pot==2));//突變點無法為頭尾
			switch (pot)//對突變點進行突變
			{
			case 0:
				population[i].degree_t[pos].Phi =fRand(0,360);
				break;
			case 1:
				population[i].degree_t[pos].Psi =fRand(0,360);
				break;
			case 2:
				population[i].degree_t[pos].W =fRand(0,360);
				break;
			default:
				break;
			}

		}	
	}
}

//---------------------Structure Transform------------------------
void RotationMatrix(double angle, double a,double b, double c, double u, double v, double w,double inputMatrix[4][1],double outputMatrix[4][1])//旋轉
{//旋轉矩陣副函式
	double rotationMatrix[4][4];

    double L = (u*u + v * v + w * w);
    angle = angle * M_PI / 180.0; //converting to radian value
    double u2 = u * u;
    double v2 = v * v;
    double w2 = w * w; 
	//旋轉矩陣計算
    rotationMatrix[0][0] = (u2 + (v2 + w2) * cos(angle)) / L;
    rotationMatrix[0][1] = (u * v * (1 - cos(angle)) - w * sqrt(L) * sin(angle)) / L;
    rotationMatrix[0][2] = (u * w * (1 - cos(angle)) + v * sqrt(L) * sin(angle)) / L;
    rotationMatrix[0][3] = ((a*(v2+w2)-u*(b*v+c*w))*(1-cos(angle))+(b*w-c*v)*sqrt(L)*sin(angle))/L;
 
    rotationMatrix[1][0] = (u * v * (1 - cos(angle)) + w * sqrt(L) * sin(angle)) / L;
    rotationMatrix[1][1] = (v2 + (u2 + w2) * cos(angle)) / L;
    rotationMatrix[1][2] = (v * w * (1 - cos(angle)) - u * sqrt(L) * sin(angle)) / L;
    rotationMatrix[1][3] = ((b*(u2+w2)-v*(a*u+c*w))*(1-cos(angle))+(c*u-a*w)*sqrt(L)*sin(angle))/L; 
 
    rotationMatrix[2][0] = (u * w * (1 - cos(angle)) - v * sqrt(L) * sin(angle)) / L;
    rotationMatrix[2][1] = (v * w * (1 - cos(angle)) + u * sqrt(L) * sin(angle)) / L;
    rotationMatrix[2][2] = (w2 + (u2 + v2) * cos(angle)) / L;
    rotationMatrix[2][3] = ((c*(u2+v2)-w*(a*u+b*v))*(1-cos(angle))+(a*v-b*u)*sqrt(L)*sin(angle))/L;  
 
    rotationMatrix[3][0] = 0.0;
    rotationMatrix[3][1] = 0.0;
    rotationMatrix[3][2] = 0.0;
    rotationMatrix[3][3] = 1.0;

	 for(int i = 0; i < 4; i++ ){
        for(int j = 0; j < 1; j++){
            outputMatrix[i][j] = 0;
            for(int k = 0; k < 4; k++){
                outputMatrix[i][j] += rotationMatrix[i][k] * inputMatrix[k][j];
            }
			if(abs(outputMatrix[i][j]) < 0.0001)
				outputMatrix[i][j]=0;
        }
    }
}
void Transform(Coordinate ***chromosome,int G_LENGTH){
	double inputMatrix[4][1] = {0.0, 0.0, 0.0, 0.0};//旋轉點
	double outputMatrix[4][1] = {0.0, 0.0, 0.0, 0.0};//旋轉後點
	double a; double b; double c;//旋轉軸通過點(a, b, c)
	double angle; double u; double v; double w;//角度、旋轉軸(u, v, w)

	for(int i=0; i<POPULATION_CNT; i++){//Chromosome第i條
		for(int j=0; j<G_LENGTH; j++){//Chromosome第i條中，第j個胺基酸
			if(population[i].degree_t[j].Phi != 0){//對Phi做旋轉
				angle = population[i].degree_t[j].Phi;//角度
				if( strcmp(chromosome[i][j][0].Aname, "PRO") == 0 ){//找旋轉軸
					a = chromosome[i][j][10].x;
					b = chromosome[i][j][10].y;
					c = chromosome[i][j][10].z;
					u = chromosome[i][j][0].x - chromosome[i][j][10].x;
					v = chromosome[i][j][0].y - chromosome[i][j][10].y;
					w = chromosome[i][j][0].z - chromosome[i][j][10].z;
				}
				else {
					a = chromosome[i][j][0].x;
					b = chromosome[i][j][0].y;
					c = chromosome[i][j][0].z;
					u = chromosome[i][j][1].x - chromosome[i][j][0].x;
					v = chromosome[i][j][1].y - chromosome[i][j][0].y;
					w = chromosome[i][j][1].z - chromosome[i][j][0].z;
				}
				for(int k=0; chromosome[i][j][k].no>0; k++){
					if(chromosome[i][j][k].group!=0){
						inputMatrix[0][0]=chromosome[i][j][k].x;//旋轉點
						inputMatrix[1][0]=chromosome[i][j][k].y;
						inputMatrix[2][0]=chromosome[i][j][k].z;
						inputMatrix[3][0]=1.0;
						RotationMatrix(angle, a, b, c, u, v, w,inputMatrix,outputMatrix);//旋轉
						chromosome[i][j][k].x = outputMatrix[0][0];//旋轉後點
						chromosome[i][j][k].y = outputMatrix[1][0];
						chromosome[i][j][k].z = outputMatrix[2][0];
					}
				}
				for(int m=j+1; m<G_LENGTH; m++){
					for(int k=0; chromosome[i][m][k].no>0; k++){
						inputMatrix[0][0]=chromosome[i][m][k].x;//旋轉點
						inputMatrix[1][0]=chromosome[i][m][k].y;
						inputMatrix[2][0]=chromosome[i][m][k].z;
						inputMatrix[3][0]=1.0;
						RotationMatrix(angle, a, b, c, u, v, w,inputMatrix,outputMatrix);//旋轉
						chromosome[i][m][k].x = outputMatrix[0][0];//旋轉後點
						chromosome[i][m][k].y = outputMatrix[1][0];
						chromosome[i][m][k].z = outputMatrix[2][0];
					}
				}
			}

			if(population[i].degree_t[j].Psi != 0){//對Psi做旋轉
				angle = population[i].degree_t[j].Psi;//角度
				if( strcmp(chromosome[i][j][0].Aname, "PRO") == 0 ){//找旋轉軸
					a = chromosome[i][j][0].x;
					b = chromosome[i][j][0].y;
					c = chromosome[i][j][0].z;
					u = chromosome[i][j][11].x - chromosome[i][j][0].x;
					v = chromosome[i][j][11].y - chromosome[i][j][0].y;
					w = chromosome[i][j][11].z - chromosome[i][j][0].z;
				}
				else {
					a = chromosome[i][j][1].x;
					b = chromosome[i][j][1].y;
					c = chromosome[i][j][1].z;
					u = chromosome[i][j][2].x - chromosome[i][j][1].x;
					v = chromosome[i][j][2].y - chromosome[i][j][1].y;
					w = chromosome[i][j][2].z - chromosome[i][j][1].z;
				}
				for(int k=0; chromosome[i][j][k].no>0; k++){
					if(chromosome[i][j][k].group==2){
						inputMatrix[0][0]=chromosome[i][j][k].x;//旋轉點
						inputMatrix[1][0]=chromosome[i][j][k].y;
						inputMatrix[2][0]=chromosome[i][j][k].z;
						inputMatrix[3][0]=1.0;
						RotationMatrix(angle, a, b, c, u, v, w,inputMatrix,outputMatrix);//旋轉
						chromosome[i][j][k].x = outputMatrix[0][0];//旋轉後點
						chromosome[i][j][k].y = outputMatrix[1][0];
						chromosome[i][j][k].z = outputMatrix[2][0];
					}
				}
				for(int m=j+1; m<G_LENGTH; m++){
					for(int k=0; chromosome[i][m][k].no>0; k++){
						inputMatrix[0][0]=chromosome[i][m][k].x;//旋轉點
						inputMatrix[1][0]=chromosome[i][m][k].y;
						inputMatrix[2][0]=chromosome[i][m][k].z;
						inputMatrix[3][0]=1.0;
						RotationMatrix(angle, a, b, c, u, v, w,inputMatrix,outputMatrix);//旋轉
						chromosome[i][m][k].x = outputMatrix[0][0];//旋轉後點
						chromosome[i][m][k].y = outputMatrix[1][0];
						chromosome[i][m][k].z = outputMatrix[2][0];
					}
				}
			}

			if(population[i].degree_t->W != 0){
				angle = population[i].degree_t[j].W;//角度
			}
		}
	}
}
void Peptidebond(double x[], double y[], double z[], double F[]){
	double a[3], b[3], c[3], L=0;//OC向量a,CAC向量b,法向量c
	double newX[3], angle=120;
	double inputMatrix[4][1] = {0.0, 0.0, 0.0, 0.0};//旋轉點
	double outputMatrix[4][1] = {0.0, 0.0, 0.0, 0.0};//旋轉後點

	for(int i=0; i<3; i++){//向量、肽鍵長計算
		a[i] = z[i]-x[i];
		b[i] = y[i]-x[i];
		L += b[i]*b[i];
	}

	c[0] = a[1]*b[2]-a[2]*b[1];//法向量
	c[1] = a[2]*b[0]-a[0]*b[2];
	c[2] = a[0]*b[1]-a[1]*b[0];

	L = sqrt(L);
	for(int i=0; i<3; i++){//鍵結點正規化
		b[i] = 1.35*b[i]/L;
		newX[i] = x[i]+b[i];
	}

	inputMatrix[0][0]=newX[0];//旋轉鍵結點至鍵結位置
	inputMatrix[1][0]=newX[1];
	inputMatrix[2][0]=newX[2];
	inputMatrix[3][0]=1.0;
	RotationMatrix(angle, x[0], x[1], x[2], c[0], c[1], c[2],inputMatrix,outputMatrix);//旋轉
	F[0] = outputMatrix[0][0];
	F[1] = outputMatrix[1][0];
	F[2] = outputMatrix[2][0];
}
void Binding_Correction(Coordinate **PredictChain, int location){
	double OC[3], HN[3], CN[3], NC[3];//OC向量, HN向量, CN向量, NC向量
	double frontNV[3], backNV[3], L1=0, L2=0;//前法向量, 後法向量, 前法向量長度, 後法向量長度
	double NV[3], x[3], angle, a;//旋轉軸, 旋轉軸通過點, 角度
	double inputMatrix[4][1] = {0.0, 0.0, 0.0, 0.0};//旋轉點
	double outputMatrix[4][1] = {0.0, 0.0, 0.0, 0.0};//旋轉後點

	//OC向量, HN向量, CN向量, NC向量
	if(strcmp(PredictChain[location-1][0].Aname,"PRO")==0 && strcmp(PredictChain[location][0].name,"PRO")==0){//PP
		OC[0] = PredictChain[location-1][13].x-PredictChain[location-1][11].x;
		OC[1] = PredictChain[location-1][13].y-PredictChain[location-1][11].y;
		OC[2] = PredictChain[location-1][13].z-PredictChain[location-1][11].z;
		HN[0] = PredictChain[location][5].x-PredictChain[location][10].x;
		HN[1] = PredictChain[location][5].y-PredictChain[location][10].y;
		HN[2] = PredictChain[location][5].z-PredictChain[location][10].z;
		CN[0] = PredictChain[location-1][11].x-PredictChain[location][10].x;
		CN[1] = PredictChain[location-1][11].y-PredictChain[location][10].y;
		CN[2] = PredictChain[location-1][11].z-PredictChain[location][10].z;
		NC[0] = PredictChain[location][10].x-PredictChain[location-1][11].x;
		NC[1] = PredictChain[location][10].y-PredictChain[location-1][11].y;
		NC[2] = PredictChain[location][10].z-PredictChain[location-1][11].z;
	}
	else if(strcmp(PredictChain[location][0].Aname,"PRO")==0){//_P
		OC[0] = PredictChain[location-1][3].x-PredictChain[location-1][2].x;
		OC[1] = PredictChain[location-1][3].y-PredictChain[location-1][2].y;
		OC[2] = PredictChain[location-1][3].z-PredictChain[location-1][2].z;
		HN[0] = PredictChain[location][5].x-PredictChain[location][10].x;
		HN[1] = PredictChain[location][5].y-PredictChain[location][10].y;
		HN[2] = PredictChain[location][5].z-PredictChain[location][10].z;
		CN[0] = PredictChain[location-1][2].x-PredictChain[location][10].x;
		CN[1] = PredictChain[location-1][2].y-PredictChain[location][10].y;
		CN[2] = PredictChain[location-1][2].z-PredictChain[location][10].z;
		NC[0] = PredictChain[location][10].x-PredictChain[location-1][2].x;
		NC[1] = PredictChain[location][10].y-PredictChain[location-1][2].y;
		NC[2] = PredictChain[location][10].z-PredictChain[location-1][2].z;
	}
	else if(strcmp(PredictChain[location-1][0].Aname,"PRO")==0){//P_
		OC[0] = PredictChain[location-1][13].x-PredictChain[location-1][11].x;
		OC[1] = PredictChain[location-1][13].y-PredictChain[location-1][11].y;
		OC[2] = PredictChain[location-1][13].z-PredictChain[location-1][11].z;
		HN[0] = PredictChain[location][4].x-PredictChain[location][0].x;
		HN[1] = PredictChain[location][4].y-PredictChain[location][0].y;
		HN[2] = PredictChain[location][4].z-PredictChain[location][0].z;
		CN[0] = PredictChain[location-1][11].x-PredictChain[location][0].x;
		CN[1] = PredictChain[location-1][11].y-PredictChain[location][0].y;
		CN[2] = PredictChain[location-1][11].z-PredictChain[location][0].z;
		NC[0] = PredictChain[location][0].x-PredictChain[location-1][11].x;
		NC[1] = PredictChain[location][0].y-PredictChain[location-1][11].y;
		NC[2] = PredictChain[location][0].z-PredictChain[location-1][11].z;
	}
	else{//__
		OC[0] = PredictChain[location-1][3].x-PredictChain[location-1][2].x;
		OC[1] = PredictChain[location-1][3].y-PredictChain[location-1][2].y;
		OC[2] = PredictChain[location-1][3].z-PredictChain[location-1][2].z;
		HN[0] = PredictChain[location][4].x-PredictChain[location][0].x;
		HN[1] = PredictChain[location][4].y-PredictChain[location][0].y;
		HN[2] = PredictChain[location][4].z-PredictChain[location][0].z;
		CN[0] = PredictChain[location-1][2].x-PredictChain[location][0].x;
		CN[1] = PredictChain[location-1][2].y-PredictChain[location][0].y;
		CN[2] = PredictChain[location-1][2].z-PredictChain[location][0].z;
		NC[0] = PredictChain[location][0].x-PredictChain[location-1][2].x;
		NC[1] = PredictChain[location][0].y-PredictChain[location-1][2].y;
		NC[2] = PredictChain[location][0].z-PredictChain[location-1][2].z;
	}
	
	//前法向量, 後法向量, 前法向量長度, 後法向量長度
	frontNV[0] = OC[1]*NC[2]-OC[2]*NC[1];
	frontNV[1] = OC[2]*NC[0]-OC[0]*NC[2];
	frontNV[2] = OC[0]*NC[1]-OC[1]*NC[0];
	backNV[0] = HN[1]*CN[2]-HN[2]*CN[1];
	backNV[1] = HN[2]*CN[0]-HN[0]*CN[2];
	backNV[2] = HN[0]*CN[1]-HN[1]*CN[0];
	L1 = sqrt(pow(frontNV[0],2)+pow(frontNV[1],2)+pow(frontNV[2],2));
	L2 = sqrt(pow(backNV[0],2)+pow(backNV[1],2)+pow(backNV[2],2));

	//angle = acos( u*v / |u|*|v| )
	a = (frontNV[0]*backNV[0]+frontNV[1]*backNV[1]+frontNV[2]*backNV[2])/(L1*L2);
	angle = 180*acos(a)/M_PI;

	//旋轉軸, 旋轉軸通過點
	NV[0] = backNV[1]*frontNV[2]-backNV[2]*frontNV[1];
	NV[1] = backNV[2]*frontNV[0]-backNV[0]*frontNV[2];
	NV[2] = backNV[0]*frontNV[1]-backNV[1]*frontNV[0];

	if(strcmp(PredictChain[location][0].Aname,"PRO")==0){
		x[0] = PredictChain[location][10].x;
		x[1] = PredictChain[location][10].y;
		x[2] = PredictChain[location][10].z;
	}
	else{
		x[0] = PredictChain[location][0].x;
		x[1] = PredictChain[location][0].y;
		x[2] = PredictChain[location][0].z;
	}

	for(int i=0; PredictChain[location][i].no>0; i++){
		inputMatrix[0][0]=PredictChain[location][i].x;
		inputMatrix[1][0]=PredictChain[location][i].y;
		inputMatrix[2][0]=PredictChain[location][i].z;
		inputMatrix[3][0]=1.0;
		RotationMatrix(angle, x[0], x[1], x[2], NV[0], NV[1], NV[2],inputMatrix,outputMatrix);//旋轉
		PredictChain[location][i].x = outputMatrix[0][0];
		PredictChain[location][i].y = outputMatrix[1][0];
		PredictChain[location][i].z = outputMatrix[2][0];
	}
}
__global__ void Distance(P_distance *dev_A_distance, int A_sum, int S_sum, Coordinate *dev_chromosome, int GENETIC_LENGTH){//原子距離
	int count1=0;
	double x2, y2, z2;
	int a=threadIdx.x;

	//printf("in cuda %d %s %lf\n",a, dev_chromosome[a].name,dev_chromosome[a].x);

	for(int b=0; b<GENETIC_LENGTH; b++){
		for(int c=0; c<30; c++){
			for(int d=b; d<GENETIC_LENGTH; d++){
				if(d==b){
					for(int e=c; e<30; e++){
						
					/*	dev_strcpy(dev_A_distance[a*S_sum+count1].A1_name, dev_chromosome[a*GENETIC_LENGTH*30+b*30+c].Aname);
						dev_strcpy(dev_A_distance[a*S_sum+count1].A2_name, dev_chromosome[a*GENETIC_LENGTH*30+d*30+e].Aname);*/
						dev_strcpy(dev_A_distance[a*S_sum+count1].A1_atom, dev_chromosome[a*GENETIC_LENGTH*30+b*30+c].name,3);
						dev_strcpy(dev_A_distance[a*S_sum+count1].A2_atom, dev_chromosome[a*GENETIC_LENGTH*30+d*30+e].name,3);
						
						/*dev_A_distance[a*S_sum+count1].A1_No=b;
						dev_A_distance[a*S_sum+count1].A2_No=d;
						dev_A_distance[a*S_sum+count1].A1_atom_No=c;
						dev_A_distance[a*S_sum+count1].A2_atom_No=e;*/
						x2 = pow(dev_chromosome[a*GENETIC_LENGTH*30+b*30+c].x-dev_chromosome[a*GENETIC_LENGTH*30+d*30+e].x,2);
						y2 = pow(dev_chromosome[a*GENETIC_LENGTH*30+b*30+c].y-dev_chromosome[a*GENETIC_LENGTH*30+d*30+e].y,2);
						z2 = pow(dev_chromosome[a*GENETIC_LENGTH*30+b*30+c].z-dev_chromosome[a*GENETIC_LENGTH*30+d*30+e].z,2);
						dev_A_distance[a*S_sum+count1].distance=sqrt(x2+y2+z2);
						count1++;
					}
				}
				else{
					for(int e=0; e<30; e++){
						
						/*dev_strcpy(dev_A_distance[a*S_sum+count1].A1_name, dev_chromosome[a*GENETIC_LENGTH*30+b*30+c].Aname);
						dev_strcpy(dev_A_distance[a*S_sum+count1].A2_name, dev_chromosome[a*GENETIC_LENGTH*30+d*30+e].Aname);*/
						dev_strcpy(dev_A_distance[a*S_sum+count1].A1_atom, dev_chromosome[a*GENETIC_LENGTH*30+b*30+c].name,3);
						dev_strcpy(dev_A_distance[a*S_sum+count1].A2_atom, dev_chromosome[a*GENETIC_LENGTH*30+d*30+e].name,3);
						/*dev_A_distance[a*S_sum+count1].A1_No=b;
						dev_A_distance[a*S_sum+count1].A2_No=d;
						dev_A_distance[a*S_sum+count1].A1_atom_No=c;
						dev_A_distance[a*S_sum+count1].A2_atom_No=e;*/
						//printf("in cuda2 %s %lf\n", dev_A_distance[a*S_sum+count1].A1_atom,dev_chromosome[a*GENETIC_LENGTH*30+b*30+c].x);
						x2 = pow(dev_chromosome[a*GENETIC_LENGTH*30+b*30+c].x-dev_chromosome[a*GENETIC_LENGTH*30+d*30+e].x,2);
						y2 = pow(dev_chromosome[a*GENETIC_LENGTH*30+b*30+c].y-dev_chromosome[a*GENETIC_LENGTH*30+d*30+e].y,2);
						z2 = pow(dev_chromosome[a*GENETIC_LENGTH*30+b*30+c].z-dev_chromosome[a*GENETIC_LENGTH*30+d*30+e].z,2);
						dev_A_distance[a*S_sum+count1].distance=sqrt(x2+y2+z2);
						count1++;
					}
				}
			}
		}
	}
}
void Find_bestgene(Coordinate ***chromosome,Coordinate **BestChain, int G_LENGTH){//搜尋最佳結構
	double bestfitness=0;

	for(int i=0; i<POPULATION_CNT; i++){
		if(i==0){//紀錄第一個結構
			bestfitness = population[i].fitness;
			memcpy(BestChain,chromosome[i],G_LENGTH*sizeof(chromosome[i]));
		}
		else if(population[i].fitness > bestfitness){//有更好的結構
			bestfitness = population[i].fitness;
			memcpy(BestChain,chromosome[i],G_LENGTH*sizeof(chromosome[i]));
		}
	}
}