#include "darknet.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <winsock2.h>
#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#endif

#include "parser.h"
#include "utils.h"
#include "dark_cuda.h"
#include "blas.h"
#include "connected_layer.h"

#pragma comment(lib, "ws2_32.lib")

#define QUITREQ		-1		// 접속 종료 요청
#define BUFSIZE 1024		// 버퍼 사이즈

// 쇼핑카트ip와 스마트폰ip를 매핑해주기 위해 딕셔너리 자료구조를 이용
typedef struct {
    char *key;
    char *value;
} KVPair;

typedef struct Dictionary_t {
    KVPair *head;
    struct Dictionary_t *tail;
} Dictionary;

Dictionary* dict_new();
void dict_add(Dictionary *dictionary, const char *key, char *value);
int dict_has(Dictionary *dictionary, const char *key);
char* dict_get(Dictionary *dictionary, const char *key);
void dict_remove(Dictionary *dictionary, const char *key);
void dict_free(Dictionary *dictionary);
typedef struct {
    SOCKET socket;
    Dictionary *dict;
    int flag;
    char *preMsg;
    //network r_net;
} ArgumentToThr;

// 쇼핑카트 ip들 저장해두기_고도화 단계에서 MAC주소로 변경예정
char scip[5][20] = { "172.20.10.9","172.20.10.9","172.20.10.9","172.20.10.9","172.20.10.9" };

int clntCnt = 0;			// 접속한 사용자 수
SOCKET clntSocks[10];		// 접속한 클라이언트의 파일 디스크립터를 저장할 배열 선언
SOCKADDR_IN clntIpArr[10];	// 접속한 클라이언트의 주소를 저장할 배열 선언
HANDLE hMutex;				// Mutex 생성시 리턴되는 핸들의 저장할 변수 선언.

extern void ErrorHandling(char *message);
extern unsigned int WINAPI HandleClient(void *arg);
extern int SendMSG(Dictionary *dict, char *message, int len, int clntCnt, unsigned clntSock, int flag, char *msg, network r_net, char *weights);

extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
extern void run_voxel(int argc, char **argv);
extern void run_yolo(int argc, char **argv, objcnt *objects, FILE *file_pointer, network r_net);
extern void run_detector(int argc, char **argv, objcnt *objects, FILE *file_pointer, network r_net, char * weights);
extern void run_coco(int argc, char **argv, objcnt *objects, FILE *file_pointer, network r_net);
extern void run_writing(int argc, char **argv);
extern void run_captcha(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_dice(int argc, char **argv);
extern void run_compare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_char_rnn(int argc, char **argv);
extern void run_vid_rnn(int argc, char **argv);
extern void run_tag(int argc, char **argv);
extern void run_cifar(int argc, char **argv);
extern void run_go(int argc, char **argv);
extern void run_art(int argc, char **argv);
extern void run_super(int argc, char **argv);

void average(int argc, char *argv[])
{
    char *cfgfile = argv[2];
    char *outfile = argv[3];
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    network sum = parse_network_cfg(cfgfile);

    char *weightfile = argv[4];
    load_weights(&sum, weightfile);

    int i, j;
    int n = argc - 5;
    for (i = 0; i < n; ++i) {
        weightfile = argv[i + 5];
        load_weights(&net, weightfile);
        for (j = 0; j < net.n; ++j) {
            layer l = net.layers[j];
            layer out = sum.layers[j];
            if (l.type == CONVOLUTIONAL) {
                int num = l.n*l.c*l.size*l.size;
                axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
                if (l.batch_normalize) {
                    axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
                    axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
                    axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
                }
            }
            if (l.type == CONNECTED) {
                axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
                axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, out.weights, 1);
            }
        }
    }
    n = n + 1;
    for (j = 0; j < net.n; ++j) {
        layer l = sum.layers[j];
        if (l.type == CONVOLUTIONAL) {
            int num = l.n*l.c*l.size*l.size;
            scal_cpu(l.n, 1. / n, l.biases, 1);
            scal_cpu(num, 1. / n, l.weights, 1);
            if (l.batch_normalize) {
                scal_cpu(l.n, 1. / n, l.scales, 1);
                scal_cpu(l.n, 1. / n, l.rolling_mean, 1);
                scal_cpu(l.n, 1. / n, l.rolling_variance, 1);
            }
        }
        if (l.type == CONNECTED) {
            scal_cpu(l.outputs, 1. / n, l.biases, 1);
            scal_cpu(l.outputs*l.inputs, 1. / n, l.weights, 1);
        }
    }
    save_weights(sum, outfile);
}

void speed(char *cfgfile, int tics)
{
    if (tics == 0) tics = 1000;
    network net = parse_network_cfg(cfgfile);
    set_batch_network(&net, 1);
    int i;
    time_t start = time(0);
    image im = make_image(net.w, net.h, net.c);
    for (i = 0; i < tics; ++i) {
        network_predict(net, im.data);
    }
    double t = difftime(time(0), start);
    printf("\n%d evals, %f Seconds\n", tics, t);
    printf("Speed: %f sec/eval\n", t / tics);
    printf("Speed: %f Hz\n", tics / t);
}

void operations(char *cfgfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    int i;
    long ops = 0;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL) {
            ops += 2l * l.n * l.size*l.size*l.c * l.out_h*l.out_w;
        }
        else if (l.type == CONNECTED) {
            ops += 2l * l.inputs * l.outputs;
        }
        else if (l.type == RNN) {
            ops += 2l * l.input_layer->inputs * l.input_layer->outputs;
            ops += 2l * l.self_layer->inputs * l.self_layer->outputs;
            ops += 2l * l.output_layer->inputs * l.output_layer->outputs;
        }
        else if (l.type == GRU) {
            ops += 2l * l.uz->inputs * l.uz->outputs;
            ops += 2l * l.uh->inputs * l.uh->outputs;
            ops += 2l * l.ur->inputs * l.ur->outputs;
            ops += 2l * l.wz->inputs * l.wz->outputs;
            ops += 2l * l.wh->inputs * l.wh->outputs;
            ops += 2l * l.wr->inputs * l.wr->outputs;
        }
        else if (l.type == LSTM) {
            ops += 2l * l.uf->inputs * l.uf->outputs;
            ops += 2l * l.ui->inputs * l.ui->outputs;
            ops += 2l * l.ug->inputs * l.ug->outputs;
            ops += 2l * l.uo->inputs * l.uo->outputs;
            ops += 2l * l.wf->inputs * l.wf->outputs;
            ops += 2l * l.wi->inputs * l.wi->outputs;
            ops += 2l * l.wg->inputs * l.wg->outputs;
            ops += 2l * l.wo->inputs * l.wo->outputs;
        }
    }
    printf("Floating Point Operations: %ld\n", ops);
    printf("Floating Point Operations: %.2f Bn\n", (float)ops / 1000000000.);
}

void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    int oldn = net.layers[net.n - 2].n;
    int c = net.layers[net.n - 2].c;
    net.layers[net.n - 2].n = 9372;
    net.layers[net.n - 2].biases += 5;
    net.layers[net.n - 2].weights += 5 * c;
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    net.layers[net.n - 2].biases -= 5;
    net.layers[net.n - 2].weights -= 5 * c;
    net.layers[net.n - 2].n = oldn;
    printf("%d\n", oldn);
    layer l = net.layers[net.n - 2];
    copy_cpu(l.n / 3, l.biases, 1, l.biases + l.n / 3, 1);
    copy_cpu(l.n / 3, l.biases, 1, l.biases + 2 * l.n / 3, 1);
    copy_cpu(l.n / 3 * l.c, l.weights, 1, l.weights + l.n / 3 * l.c, 1);
    copy_cpu(l.n / 3 * l.c, l.weights, 1, l.weights + 2 * l.n / 3 * l.c, 1);
    *net.seen = 0;
    save_weights(net, outfile);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights_upto(&net, weightfile, max);
    }
    *net.seen = 0;
    save_weights_upto(net, outfile, max);
}

#include "convolutional_layer.h"
void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL) {
            rescale_weights(l, 2, -.5);
            break;
        }
    }
    save_weights(net, outfile);
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL) {
            rgbgr_weights(l);
            break;
        }
    }
    save_weights(net, outfile);
}

void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize) {
            denormalize_convolutional_layer(l);
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
        }
        if (l.type == LSTM && l.batch_normalize) {
            denormalize_connected_layer(*l.wf);
            denormalize_connected_layer(*l.wi);
            denormalize_connected_layer(*l.wg);
            denormalize_connected_layer(*l.wo);
            denormalize_connected_layer(*l.uf);
            denormalize_connected_layer(*l.ui);
            denormalize_connected_layer(*l.ug);
            denormalize_connected_layer(*l.uo);
        }
    }
    save_weights(net, outfile);
}

layer normalize_layer(layer l, int n)
{
    int j;
    l.batch_normalize = 1;
    l.scales = (float*)calloc(n, sizeof(float));
    for (j = 0; j < n; ++j) {
        l.scales[j] = 1;
    }
    l.rolling_mean = (float*)calloc(n, sizeof(float));
    l.rolling_variance = (float*)calloc(n, sizeof(float));
    return l;
}

void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL && !l.batch_normalize) {
            net.layers[i] = normalize_layer(l, l.n);
        }
        if (l.type == CONNECTED && !l.batch_normalize) {
            net.layers[i] = normalize_layer(l, l.outputs);
        }
        if (l.type == GRU && l.batch_normalize) {
            *l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
            *l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
            *l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
            *l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
            *l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
            *l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
            net.layers[i].batch_normalize = 1;
        }
        if (l.type == LSTM && l.batch_normalize) {
            *l.wf = normalize_layer(*l.wf, l.wf->outputs);
            *l.wi = normalize_layer(*l.wi, l.wi->outputs);
            *l.wg = normalize_layer(*l.wg, l.wg->outputs);
            *l.wo = normalize_layer(*l.wo, l.wo->outputs);
            *l.uf = normalize_layer(*l.uf, l.uf->outputs);
            *l.ui = normalize_layer(*l.ui, l.ui->outputs);
            *l.ug = normalize_layer(*l.ug, l.ug->outputs);
            *l.uo = normalize_layer(*l.uo, l.uo->outputs);
            net.layers[i].batch_normalize = 1;
        }
    }
    save_weights(net, outfile);
}

void statistics_net(char *cfgfile, char *weightfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONNECTED && l.batch_normalize) {
            printf("Connected Layer %d\n", i);
            statistics_connected_layer(l);
        }
        if (l.type == GRU && l.batch_normalize) {
            printf("GRU Layer %d\n", i);
            printf("Input Z\n");
            statistics_connected_layer(*l.input_z_layer);
            printf("Input R\n");
            statistics_connected_layer(*l.input_r_layer);
            printf("Input H\n");
            statistics_connected_layer(*l.input_h_layer);
            printf("State Z\n");
            statistics_connected_layer(*l.state_z_layer);
            printf("State R\n");
            statistics_connected_layer(*l.state_r_layer);
            printf("State H\n");
            statistics_connected_layer(*l.state_h_layer);
        }
        if (l.type == LSTM && l.batch_normalize) {
            printf("LSTM Layer %d\n", i);
            printf("wf\n");
            statistics_connected_layer(*l.wf);
            printf("wi\n");
            statistics_connected_layer(*l.wi);
            printf("wg\n");
            statistics_connected_layer(*l.wg);
            printf("wo\n");
            statistics_connected_layer(*l.wo);
            printf("uf\n");
            statistics_connected_layer(*l.uf);
            printf("ui\n");
            statistics_connected_layer(*l.ui);
            printf("ug\n");
            statistics_connected_layer(*l.ug);
            printf("uo\n");
            statistics_connected_layer(*l.uo);
        }
        printf("\n");
    }
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{
    gpu_index = -1;
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    int i;
    for (i = 0; i < net.n; ++i) {
        layer l = net.layers[i];
        if (l.type == CONVOLUTIONAL && l.batch_normalize) {
            denormalize_convolutional_layer(l);
            net.layers[i].batch_normalize = 0;
        }
        if (l.type == CONNECTED && l.batch_normalize) {
            denormalize_connected_layer(l);
            net.layers[i].batch_normalize = 0;
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.input_z_layer);
            denormalize_connected_layer(*l.input_r_layer);
            denormalize_connected_layer(*l.input_h_layer);
            denormalize_connected_layer(*l.state_z_layer);
            denormalize_connected_layer(*l.state_r_layer);
            denormalize_connected_layer(*l.state_h_layer);
            l.input_z_layer->batch_normalize = 0;
            l.input_r_layer->batch_normalize = 0;
            l.input_h_layer->batch_normalize = 0;
            l.state_z_layer->batch_normalize = 0;
            l.state_r_layer->batch_normalize = 0;
            l.state_h_layer->batch_normalize = 0;
            net.layers[i].batch_normalize = 0;
        }
        if (l.type == GRU && l.batch_normalize) {
            denormalize_connected_layer(*l.wf);
            denormalize_connected_layer(*l.wi);
            denormalize_connected_layer(*l.wg);
            denormalize_connected_layer(*l.wo);
            denormalize_connected_layer(*l.uf);
            denormalize_connected_layer(*l.ui);
            denormalize_connected_layer(*l.ug);
            denormalize_connected_layer(*l.uo);
            l.wf->batch_normalize = 0;
            l.wi->batch_normalize = 0;
            l.wg->batch_normalize = 0;
            l.wo->batch_normalize = 0;
            l.uf->batch_normalize = 0;
            l.ui->batch_normalize = 0;
            l.ug->batch_normalize = 0;
            l.uo->batch_normalize = 0;
            net.layers[i].batch_normalize = 0;
        }
    }
    save_weights(net, outfile);
}

void visualize(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    visualize_network(net);
#ifdef OPENCV
    wait_until_press_key_cv();
#endif
}

int main(int argc, char **argv)
{
#ifdef _DEBUG
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    // 각 작업 스레드로 넘길 데이터들을 구조체로 모아주기 위해 동적할당
    ArgumentToThr *argToThreads = (ArgumentToThr*)malloc(sizeof(ArgumentToThr));
    argToThreads->dict = dict_new();
    argToThreads->flag = 1;
    argToThreads->preMsg = (char*)malloc(sizeof(char)*BUFSIZE);
    for (int x = 0; x < BUFSIZE; x++) {
        argToThreads->preMsg[x] = 0;
    }
    //argToThreads->r_net = parse_network_cfg_custom("data/yolo-obj.cfg", 1, 1);
    // 서버와 통신을 위하여 winsock을 사용한다.
    WSADATA wsaData;
    SOCKET servSock, clntSock;

    // 주소 정보를 담는 구조체 변수
    SOCKADDR_IN servAddr;
    SOCKADDR_IN clntAddr;

    HANDLE hThread;		// 스레드 핸들
    DWORD dwThreadID;	// 스레드 ID

    char clntAddrsize = sizeof(SOCKADDR_IN);					// 클라이언트 주소 저장할 변수
    char message[BUFSIZE] = { "server connected...\n" };		// 메세지 버퍼
    int strLen;
    int clntAddrSize = sizeof(SOCKADDR_IN);
    int nRcv;

    // 프로그램 시작 시 포트넘버를 입력받지 않은 경우
    
    if (argc != 2)
    {
        printf("Please, Insert Port Number\n");
        exit(1);
    }

    // 윈도우 소켓을 초기화시킨다.
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        ErrorHandling("Load WinSock 2.2DLL Error");
    }

    // 동기화 처리를 위한 뮤텍스 핸들 생성
    if ((hMutex = CreateMutex(NULL, FALSE, NULL)) == NULL) {
        ErrorHandling("CreateMutex() Error");
    }

    // 클라이언트의 통신 요청에 대한 listening용도의 서버 소켓을 생성한다.(IPv4, SOCK_STREAM 설정)
    if ((servSock = socket(PF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        ErrorHandling("Socket Error");
    }

    // 서버 주소를 설정한다.
    memset(&servAddr, 0, sizeof(SOCKADDR_IN));
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = htons(atoi(argv[1]));	// 프로그램 시작 시 인수로 받았던 문자열을 포트넘버로 지정

    // 설정된 서버주소로 서비스 개시
    if (bind(servSock, (void *)&servAddr, sizeof(servAddr)) == SOCKET_ERROR) {
        ErrorHandling("Bind Error");
    }

    // 클라이언트의 연결을 받을때까지 listen하기
    if (listen(servSock, 2) == SOCKET_ERROR) {
        ErrorHandling("Listen Error");
    }
#ifndef GPU
    gpu_index = -1;
#else
    if (gpu_index >= 0) {
        cuda_set_device(gpu_index);
        CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    }
#endif
    printf("start");
    // 메인 스레드는 이제부터 언제 어떤 클라이언트가 접속을 시도할지 모르니 listen만 전담한다
    while (1) {
        // 클라이언트 연결을 수락한다.
        if ((clntSock = accept(servSock, (SOCKADDR*)&clntAddr, &clntAddrSize)) == INVALID_SOCKET) {
            ErrorHandling("Accept Error");
        }
        else
        {
            argToThreads->socket = clntSock;

            // 크리티컬 섹션_클라이언트 수 증가시키고 각 소켓과 클라이언트 ip address 저장
            WaitForSingleObject(hMutex, INFINITE);
            {
                clntSocks[clntCnt] = clntSock;
                clntIpArr[clntCnt++] = clntAddr;
            }
            ReleaseMutex(hMutex);
            // 크리티컬 섹션 끝

            // 서버에 클라이언트의 접속을 알린다
            printf("%s Connection of New Client Complete!\n", inet_ntoa(clntAddr.sin_addr));

            // 새 HandleClient 작업 스레드를 시작_각 클라이언트 소켓마다 스레드를 갖게 된다
            hThread = (HANDLE)_beginthreadex(NULL, 0, HandleClient, (void*)argToThreads, 0, (unsigned *)&dwThreadID);

            // 스레드가 정상적으로 생성되지 않는 경우 에러 처리
            if (hThread == 0) {
                ErrorHandling("_beginthreadex() Error");
            }
        }
    }
    // HandleClient 스레드가 작업을 종료할 때까지 기다린다.
    WaitForSingleObject(hThread, INFINITE);
    // 소켓 닫기, wsa 비우기

    closesocket(servSock);
    WSACleanup();
    return 0;
}

/* 클라이언트 핸들
* 서버와 통신할 소켓(clntSock)과
* 쇼핑카트 클라이언트와 스마트폰 클라이언트의 쌍을 표현하기 위해 구조화된 데이터인
* dict를 함께 인수로 받는다
* _스레드 생성 함수의 세번째 인자*/
unsigned int WINAPI HandleClient(void *arg)
{
    ArgumentToThr *tmpArg = (ArgumentToThr*)arg;
    Dictionary *dict = tmpArg->dict;
    SOCKET clntSock = tmpArg->socket;
    int statusFlag = tmpArg->flag;
    char *preMsg = tmpArg->preMsg;
    //network r_net = tmpArg->r_net;
    network r_net = parse_network_cfg_custom("data/yolo-obj.cfg", 1, 1);
    char *weights = "yolo-obj_last.weights";
    if (weights) {
        load_weights(&r_net, weights);
    }
    char message[BUFSIZE];		// 수신 버퍼
    int i;
    int recvLen;
    int retVal;

    // 클라이언트가 종료 요청을 하면 while문을 빠져나온다.
    while ((recvLen = recv(clntSock, message, BUFSIZE - 1, 0)) != 0) {
        // recieve Error 처리
        if (recvLen == SOCKET_ERROR) {
            printf("Receive Error..\n");
            break;
        }

        // 수신 메시지 문자열 형식 맞추기
        message[recvLen] = '\0';
        printf("input msg from client : %s\n", message);
        retVal = SendMSG(dict, message, recvLen, clntCnt, clntSock, statusFlag, preMsg, r_net, weights);
        printf("retVal : %d\n", retVal);
        // 종료 요청 메시지인지 확인
        if (retVal == QUITREQ) {
            printf("Close Client Connection..\n");
            break;
        }
        statusFlag *= -1;

        // 종료 메시지가 아니면 출력
        //printf("%s\n", message);
/*		for (int i = 0; i < BUFSIZE; i++) {
            message[i] = '\0';	// 한번 보내고나면 문자열 비우기
        }*/
        Sleep(100);	// 일정 간격을 두고 반복
    }

    SOCKET tmpAndroidClntSock = NULL;
    // 크리티컬 섹션
    WaitForSingleObject(hMutex, INFINITE);
    {
        int boool = 0;
        char * tmpAndroidClntIp = NULL;

        for (i = 0; i < clntCnt; i++) {
            if (clntSock == clntSocks[i]) {
                for (int j = 0; j < 5; j++) {	// 쇼핑카트 ip 5개 중에서
                    if (strcmp(inet_ntoa(clntIpArr[i].sin_addr), scip[j]) == 0) {	// 같은게 있다면
                        boool = 1;	// 지금 접속 종료한 클라이언트는 쇼핑카트이다
                        // 이 쇼핑카트와 짝인 스마트폰의 ip를 미리 담아두고
                        tmpAndroidClntIp = dict_get(dict, inet_ntoa(clntIpArr[i].sin_addr));
                        // 이제 쇼핑카트랑 스마트폰이랑 짝도 끊고
                        dict_remove(dict, inet_ntoa(clntIpArr[i].sin_addr));
                        break;
                    }
                }
            }
        }
        // 접속 종료한 클라이언트를 클라이언트 배열에서 제거하면서 이후 클라이언트 정보를 앞으로 당긴다.
        for (i = 0; i < clntCnt; i++) {
            if (clntSock == clntSocks[i]) {
                for (; i < clntCnt - 1; i++) {
                    printf("%d\n", clntCnt);
                    clntSocks[i] = clntSocks[i + 1];
                    clntIpArr[i] = clntIpArr[i + 1];
                }

                break;
            }
        }
        clntCnt--;

        if (boool) {	// 접속 종료한 클라이언트가 쇼핑카트였을 경우에는 스마트폰 접속도 같이 끊어준다
            // 스마트폰 소켓도 찾아낸다
            for (i = 0; i < clntCnt; i++) {
                if (strcmp(tmpAndroidClntIp, inet_ntoa(clntIpArr[i].sin_addr)) == 0) {
                    tmpAndroidClntSock = clntSocks[i];
                    // 접속 종료한 클라이언트를 클라이언트 배열에서 제거, 앞으로 당긴다.
                    for (; i < clntCnt - 1; i++) {
                        clntSocks[i] = clntSocks[i + 1];
                        clntIpArr[i] = clntIpArr[i + 1];
                    }

                    break;
                }
            }
            clntCnt--;
            free_network(r_net);
        }
    }
    ReleaseMutex(hMutex);
    // 크리티컬 섹션 끝.

    // 종료한 클라이언트 소켓 닫기
    closesocket(clntSock);
    if (tmpAndroidClntSock != NULL) {
        closesocket(tmpAndroidClntSock);
    }
    return 0;
}


/* 어느 클라이언트로부터 왔으며, 어떤 메시지인지를 확인하고
* 클라이언트에게 메시지를 전송하거나 object detection 등의 특정 동작을 하도록 설계
* 인자 : char *message		- 전송할 메시지
*        int len			- 메시지의 길이
*		 int clntCnt		- 접속 사용자의 수
*        unsigned clntSock - 요청한 사용자의 소켓 디스크립터
*/
int SendMSG(Dictionary *dict, char *message, int len, int clntCnt, unsigned clntSock, int flag, char *msg, network r_net, char *weights)
{
    char sendBuf[BUFSIZE];		// 송신 버퍼
    printf("%d flag:::::::::::", flag);

    int i, j;
    FILE *file_pointer;
    //file_pointer = fopen("C:\\Users\\JunSeop\\Desktop\\stat\\log.txt", "a+");   ////////////////////////////////////////////////////////////////// 다른 호스트에서 할 때 이부분 꼭 !바꾸기
    file_pointer = fopen("D:\\log.txt", "a+");
    objcnt objects[5];
    for (int k = 0; k < 5; k++)
        objects[k].cnt = 0;
    strcpy(objects[0].name, "coca-cola");
    strcpy(objects[1].name, "butter_ring");
    strcpy(objects[2].name, "ramen");
    strcpy(objects[3].name, "downy");
    strcpy(objects[4].name, "corn-frost");
    
    // 크리티컬 섹션
    WaitForSingleObject(hMutex, INFINITE);
    {
        // 종료 요청 시
        if (strcmp(message, "exit") == 0)
        {
            ReleaseMutex(hMutex);
            // 크리티컬 섹션 끝.
            return QUITREQ;
        }

        int boool = 0;
        // 아직 연결된 짝이 없다면
        for (i = 0; i < clntCnt; i++) {	// 클라이언트 ip들을 다 확인해볼건데
            if (clntSock == clntSocks[i]) {	// 이 클라이언트에 대해
                printf("\n\ncurrent client num : %d\n\n\n", i);
                for (j = 0; j < 5; j++) {	// 쇼핑카트 ip 5개 중에서
                    if (strcmp(inet_ntoa(clntIpArr[i].sin_addr), scip[j]) == 0) {	// 같은게 있다면
                        boool = 1;	// 이 메시지 송신인은 쇼핑카트이다.
                        break;
                    }
                }
                if (!(boool)) {	// 송신인이 쇼핑카트가 아니고 스마트폰 ip일 경우
                    // 메모리상에서 사전형태로 쇼핑카트와 스마트폰을 짝 지어준다.
                    // 스마트폰 클라이언트에서는 짝지을때만 최초 한번 메시지를 송신할 것임
                    int isIpRegex = 0;
                    // 접속된 클라이언트들의 ip 중 하나로 연결하라는 요청인지 아닌지 먼저 판별
                    //printf("%dsss\n", clntCnt);
                    for (j = 0; j < clntCnt; j++) {
                        //printf("A : %sl\nB : %sl\n%d***\n",message, inet_ntoa(clntIpArr[j].sin_addr), strcmp(message, inet_ntoa(clntIpArr[j].sin_addr)));
                        if (strcmp(message, inet_ntoa(clntIpArr[j].sin_addr)) == 0) {
                            isIpRegex = 1;
                        }
                    }
                    printf("Check %d AND %d\n", isIpRegex, dict_has(dict, message));
                    // ip 매핑요청이 맞으며, 아직 딕셔너리 체인 내에 이 쇼핑카트가 키로서 저장되어있지 않으면
                    if ((isIpRegex) && (!(dict_has(dict, message)))) {
                        printf("mapping\n\n\n");
                        dict_add(dict, message, inet_ntoa(clntIpArr[i].sin_addr));	// 딕셔너리 체인 추가
                    }
                    // 그렇지 않은 경우
                    else {
                        /*for (int i = 0; i < BUFSIZE; i++) {
                            message[i] = '\0';	// 한번 보내고나면 문자열 비우기
                        }*/
                        strcpy(message, "bad request\n");
                        send(clntSock, message, strlen(message), 0);
                    }
                }
                else {	// 송신인이 쇼핑카트일 경우
                    if (flag == 1) {
                        // check the message(str) and call the appropriate function
                        //get streaming video via opencv from url that contains hostname as if corresponding ip address of the shopping cart
                        // classificate which products these are and check the number of each products via object detection
                        //int i;
                        int argc = 0;
                        char **argv = (char **)malloc(sizeof(char *) * 7);
                    
                        for (int p = 0; p < 7; p++)
                        {
                            char *tmp = (char *)malloc(sizeof(char) * 100);
                            strcpy(tmp, "");
                            argv[p] = tmp;
                        }
                        // raspi로부터 수신된 문자열을 공백 기준으로 잘라서 명령행의 개수 argc(변수 바꾸세요)를 count, argv문자열 배열 만들기
                        char tmpMSG[100] = { 0, };
                        strcpy(tmpMSG, message);
                        char *tmpStrForCnt = strtok(tmpMSG, " ");

                        while (tmpStrForCnt != NULL) {
                            //printf("[-%s]",)
                            strcpy(argv[argc], tmpStrForCnt);
                            argc++;
                            tmpStrForCnt = strtok(NULL, " ");
                        }
                        for (int p = 0; p< argc; ++p) {
                            if (!argv[p]) continue;
                            strip_args(argv[p]);
                        }
                        
                        if (argc < 2) {
                            if (argv[0] != '+' && argv[0] != '-' && argv[0] != '=') {
                            fprintf(stderr, "usage: %s <function>\n", argv[0]);
                            return 0;
                            }
                        }
                    
                        gpu_index = find_int_arg(argc, argv, "-i", 0);
                        printf("%d", gpu_index);
                        printf("%d", find_arg(argc, argv, "-nogpu"));
                        if (find_arg(argc, argv, "-nogpu")) {
                            gpu_index = -1;
                            printf("\n Currently Darknet doesn't support -nogpu flag. If you want to use CPU - please compile Darknet with GPU=0 in the Makefile, or compile darknet_no_gpu.sln on Windows.\n");
                            exit(-1);
                        }

                        if (0 == strcmp(argv[1], "average")) {
                            average(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "yolo")) {
                            run_yolo(argc, argv, objects, file_pointer, r_net);
                        }
                        else if (0 == strcmp(argv[1], "voxel")) {
                            run_voxel(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "super")) {
                            run_super(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "detector")) {
                            run_detector(argc, argv, objects, file_pointer, r_net, weights);
                            char p1[20];
                            char p2[20];
                            
                            // 메시지 비워준다
                            for (int x = 0; x < BUFSIZE; x++) {
                                msg[x] = 0;
                            }
                            // 새로운 메시지를 넣어준다.
                            for (int z = 0; z < 5; z++)
                            {
                                sprintf(p1, "%d", objects[z].cnt);
                                sprintf(p2, "%d", z + 1);
                                fputs(objects[z].name, file_pointer);
                                fputs("객체는 ", file_pointer);
                                fputs(p1, file_pointer);
                                fputs("개 있습니다.\n", file_pointer);

                                if (objects[z].cnt == 0)
                                    continue;
                                
                                strcat(msg, p2);
                                strcat(msg, "@");
                                strcat(msg, p1);
                                strcat(msg, "@");
                                strcat(msg, ";");
                                
                            }
                            fputs("=================================\n", file_pointer);


                        }
                        else if (0 == strcmp(argv[1], "detect")) {
                            float thresh = find_float_arg(argc, argv, "-thresh", .24);
                            int ext_output = find_arg(argc, argv, "-ext_output");
                            char *filename = (argc > 4) ? argv[4] : 0;
                            test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, 0.5, 0, ext_output, 0, NULL, 0);
                        }
                        else if (0 == strcmp(argv[1], "cifar")) {
                            run_cifar(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "go")) {
                            run_go(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "rnn")) {
                            run_char_rnn(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "vid")) {
                            run_vid_rnn(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "coco")) {
                            run_coco(argc, argv, objects, file_pointer, r_net);
                        }
                        else if (0 == strcmp(argv[1], "classify")) {
                            predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
                        }
                        else if (0 == strcmp(argv[1], "classifier")) {
                            run_classifier(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "art")) {
                            run_art(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "tag")) {
                            run_tag(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "compare")) {
                            run_compare(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "dice")) {
                            run_dice(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "writing")) {
                            run_writing(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "3d")) {
                            composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
                        }
                        else if (0 == strcmp(argv[1], "test")) {
                            test_resize(argv[2]);
                        }
                        else if (0 == strcmp(argv[1], "captcha")) {
                            run_captcha(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "nightmare")) {
                            run_nightmare(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "rgbgr")) {
                            rgbgr_net(argv[2], argv[3], argv[4]);
                        }
                        else if (0 == strcmp(argv[1], "reset")) {
                            reset_normalize_net(argv[2], argv[3], argv[4]);
                        }
                        else if (0 == strcmp(argv[1], "denormalize")) {
                            denormalize_net(argv[2], argv[3], argv[4]);
                        }
                        else if (0 == strcmp(argv[1], "statistics")) {
                            statistics_net(argv[2], argv[3]);
                        }
                        else if (0 == strcmp(argv[1], "normalize")) {
                            normalize_net(argv[2], argv[3], argv[4]);
                        }
                        else if (0 == strcmp(argv[1], "rescale")) {
                            rescale_net(argv[2], argv[3], argv[4]);
                        }
                        else if (0 == strcmp(argv[1], "ops")) {
                            operations(argv[2]);
                        }
                        else if (0 == strcmp(argv[1], "speed")) {
                            speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);
                        }
                        else if (0 == strcmp(argv[1], "oneoff")) {
                            oneoff(argv[2], argv[3], argv[4]);
                        }
                        else if (0 == strcmp(argv[1], "partial")) {
                            partial(argv[2], argv[3], argv[4], atoi(argv[5]));
                        }
                        else if (0 == strcmp(argv[1], "average")) {
                            average(argc, argv);
                        }
                        else if (0 == strcmp(argv[1], "visualize")) {
                            visualize(argv[2], (argc > 3) ? argv[3] : 0);
                        }
                        else if (0 == strcmp(argv[1], "imtest")) {
                            test_resize(argv[2]);
                        }
                        else {
                            fprintf(stderr, "Not an option: %s\n", argv[1]);
                        }
                        //return 0;
                        
                        
                        fclose(file_pointer);
                        
                        printf("1path end\n");
                    }
                    else {
                        // change serialized datas to message(str)
                        char * destClnt = dict_get(dict, inet_ntoa(clntIpArr[i].sin_addr));
                        printf("[hello]");
                        printf("[%s]\n", destClnt);
                        for (j = 0; j < clntCnt; j++) {
                            // 짝 스마트폰 소켓 찾아서
                            if (strcmp(inet_ntoa(clntIpArr[j].sin_addr), destClnt) == 0) {
                                break;
                            }
                        }
                        if (j >= 0 && j < clntCnt) {
                            if (strcmp(message, "=") != 0) {    // 무게 변화가 있는 경우에만
                                strcat(msg, message);   // 이전 메시지에 현재 메시지를 추가하고 (+ or -)
                                strcat(msg, ";\n");      // 개행을 추가하고
                                strcpy(message, msg);	// test case
                                printf("sendmsg....: %sdddd", message);
                                send(clntSocks[j], message, strlen(message), 0);	// 짝 스마트폰에게 메세지를 전송해
                            }
                        }
                        
                    }
                }
            }
        }
    }
    ReleaseMutex(hMutex);
    // 크리티컬 섹션 끝.
    return 0;
}


// 에러 핸들링 함수
void ErrorHandling(char *message)
{
    printf("1");
    WSACleanup();
    fputs(message, stderr);
    fputc('\n', stderr);
    _getch();
    exit(1);
}

#include <assert.h>
#include <string.h>
#include <stdlib.h>

Dictionary* dict_new() {
    Dictionary *dictionary = (Dictionary*)malloc(sizeof(Dictionary));
    assert(dictionary != NULL);
    dictionary->head = NULL;
    dictionary->tail = NULL;
    return dictionary;
}

void dict_add(Dictionary *dictionary, const char *key, char *value) {
    if (dict_has(dictionary, key))
        dict_remove(dictionary, key);
    if (dictionary->head != NULL) {
        while (dictionary->tail != NULL) {
            dictionary = dictionary->tail;
        }
        Dictionary *next = dict_new();
        dictionary->tail = next;
        dictionary = dictionary->tail;
    }
    int key_length = strlen(key) + 1;
    int value_length = strlen(value) + 1;
    dictionary->head = (KVPair*)malloc(sizeof(KVPair));
    assert(dictionary->head != NULL);
    dictionary->head->key = (char *)malloc(key_length * sizeof(char));
    assert(dictionary->head->key != NULL);	//	key값은 null이면 안된다
    strcpy(dictionary->head->key, key);
    dictionary->head->value = (char *)malloc(value_length * sizeof(char));
    strcpy(dictionary->head->value, value);

}

int dict_has(Dictionary *dictionary, const char *key) {
    if (dictionary->head == NULL)
        return 0;
    while (dictionary != NULL) {
        if (strcmp(dictionary->head->key, key) == 0)
            return 1;
        dictionary = dictionary->tail;
    }
    return 0;
}

char* dict_get(Dictionary *dictionary, const char *key) {
    if (dictionary->head == NULL)
        return 0;
    while (dictionary != NULL) {
        if (strcmp(dictionary->head->key, key) == 0)
            return dictionary->head->value;
        printf("checkdict %sddd, %sddd\n", dictionary->head->key, key);
        dictionary = dictionary->tail;
    }
    return 0;
}

void dict_remove(Dictionary *dictionary, const char *key) {
    if (dictionary->head == NULL)
        return;
    Dictionary *previous = NULL;
    while (dictionary != NULL) {
        if (strcmp(dictionary->head->key, key) == 0) {
            if (previous == NULL) {
                free(dictionary->head->key);
                free(dictionary->head->value);
                dictionary->head->key = NULL;
                dictionary->head->value = NULL;
                if (dictionary->tail != NULL) {
                    Dictionary *toremove = dictionary->tail;
                    dictionary->head->key = toremove->head->key;
                    dictionary->head->value = toremove->head->value;
                    dictionary->tail = toremove->tail;
                    free(toremove->head);
                    free(toremove);
                    return;
                }
            }
            else {
                previous->tail = dictionary->tail;
            }
            free(dictionary->head->key);
            free(dictionary->head->value);
            free(dictionary->head);
            free(dictionary);
            return;
        }
        previous = dictionary;
        dictionary = dictionary->tail;
    }
}

void dict_free(Dictionary *dictionary) {
    if (dictionary == NULL)
        return;
    free(dictionary->head->key);
    free(dictionary->head->value);
    free(dictionary->head);
    Dictionary *tail = dictionary->tail;
    free(dictionary);
    dict_free(tail);
}