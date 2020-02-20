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

#define QUITREQ		-1		// ���� ���� ��û
#define BUFSIZE 1024		// ���� ������

// ����īƮip�� ����Ʈ��ip�� �������ֱ� ���� ��ųʸ� �ڷᱸ���� �̿�
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

// ����īƮ ip�� �����صα�_��ȭ �ܰ迡�� MAC�ּҷ� ���濹��
char scip[5][20] = { "172.20.10.9","172.20.10.9","172.20.10.9","172.20.10.9","172.20.10.9" };

int clntCnt = 0;			// ������ ����� ��
SOCKET clntSocks[10];		// ������ Ŭ���̾�Ʈ�� ���� ��ũ���͸� ������ �迭 ����
SOCKADDR_IN clntIpArr[10];	// ������ Ŭ���̾�Ʈ�� �ּҸ� ������ �迭 ����
HANDLE hMutex;				// Mutex ������ ���ϵǴ� �ڵ��� ������ ���� ����.

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

    // �� �۾� ������� �ѱ� �����͵��� ����ü�� ����ֱ� ���� �����Ҵ�
    ArgumentToThr *argToThreads = (ArgumentToThr*)malloc(sizeof(ArgumentToThr));
    argToThreads->dict = dict_new();
    argToThreads->flag = 1;
    argToThreads->preMsg = (char*)malloc(sizeof(char)*BUFSIZE);
    for (int x = 0; x < BUFSIZE; x++) {
        argToThreads->preMsg[x] = 0;
    }
    //argToThreads->r_net = parse_network_cfg_custom("data/yolo-obj.cfg", 1, 1);
    // ������ ����� ���Ͽ� winsock�� ����Ѵ�.
    WSADATA wsaData;
    SOCKET servSock, clntSock;

    // �ּ� ������ ��� ����ü ����
    SOCKADDR_IN servAddr;
    SOCKADDR_IN clntAddr;

    HANDLE hThread;		// ������ �ڵ�
    DWORD dwThreadID;	// ������ ID

    char clntAddrsize = sizeof(SOCKADDR_IN);					// Ŭ���̾�Ʈ �ּ� ������ ����
    char message[BUFSIZE] = { "server connected...\n" };		// �޼��� ����
    int strLen;
    int clntAddrSize = sizeof(SOCKADDR_IN);
    int nRcv;

    // ���α׷� ���� �� ��Ʈ�ѹ��� �Է¹��� ���� ���
    
    if (argc != 2)
    {
        printf("Please, Insert Port Number\n");
        exit(1);
    }

    // ������ ������ �ʱ�ȭ��Ų��.
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        ErrorHandling("Load WinSock 2.2DLL Error");
    }

    // ����ȭ ó���� ���� ���ؽ� �ڵ� ����
    if ((hMutex = CreateMutex(NULL, FALSE, NULL)) == NULL) {
        ErrorHandling("CreateMutex() Error");
    }

    // Ŭ���̾�Ʈ�� ��� ��û�� ���� listening�뵵�� ���� ������ �����Ѵ�.(IPv4, SOCK_STREAM ����)
    if ((servSock = socket(PF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET) {
        ErrorHandling("Socket Error");
    }

    // ���� �ּҸ� �����Ѵ�.
    memset(&servAddr, 0, sizeof(SOCKADDR_IN));
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = htons(atoi(argv[1]));	// ���α׷� ���� �� �μ��� �޾Ҵ� ���ڿ��� ��Ʈ�ѹ��� ����

    // ������ �����ּҷ� ���� ����
    if (bind(servSock, (void *)&servAddr, sizeof(servAddr)) == SOCKET_ERROR) {
        ErrorHandling("Bind Error");
    }

    // Ŭ���̾�Ʈ�� ������ ���������� listen�ϱ�
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
    // ���� ������� �������� ���� � Ŭ���̾�Ʈ�� ������ �õ����� �𸣴� listen�� �����Ѵ�
    while (1) {
        // Ŭ���̾�Ʈ ������ �����Ѵ�.
        if ((clntSock = accept(servSock, (SOCKADDR*)&clntAddr, &clntAddrSize)) == INVALID_SOCKET) {
            ErrorHandling("Accept Error");
        }
        else
        {
            argToThreads->socket = clntSock;

            // ũ��Ƽ�� ����_Ŭ���̾�Ʈ �� ������Ű�� �� ���ϰ� Ŭ���̾�Ʈ ip address ����
            WaitForSingleObject(hMutex, INFINITE);
            {
                clntSocks[clntCnt] = clntSock;
                clntIpArr[clntCnt++] = clntAddr;
            }
            ReleaseMutex(hMutex);
            // ũ��Ƽ�� ���� ��

            // ������ Ŭ���̾�Ʈ�� ������ �˸���
            printf("%s Connection of New Client Complete!\n", inet_ntoa(clntAddr.sin_addr));

            // �� HandleClient �۾� �����带 ����_�� Ŭ���̾�Ʈ ���ϸ��� �����带 ���� �ȴ�
            hThread = (HANDLE)_beginthreadex(NULL, 0, HandleClient, (void*)argToThreads, 0, (unsigned *)&dwThreadID);

            // �����尡 ���������� �������� �ʴ� ��� ���� ó��
            if (hThread == 0) {
                ErrorHandling("_beginthreadex() Error");
            }
        }
    }
    // HandleClient �����尡 �۾��� ������ ������ ��ٸ���.
    WaitForSingleObject(hThread, INFINITE);
    // ���� �ݱ�, wsa ����

    closesocket(servSock);
    WSACleanup();
    return 0;
}

/* Ŭ���̾�Ʈ �ڵ�
* ������ ����� ����(clntSock)��
* ����īƮ Ŭ���̾�Ʈ�� ����Ʈ�� Ŭ���̾�Ʈ�� ���� ǥ���ϱ� ���� ����ȭ�� ��������
* dict�� �Բ� �μ��� �޴´�
* _������ ���� �Լ��� ����° ����*/
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
    char message[BUFSIZE];		// ���� ����
    int i;
    int recvLen;
    int retVal;

    // Ŭ���̾�Ʈ�� ���� ��û�� �ϸ� while���� �������´�.
    while ((recvLen = recv(clntSock, message, BUFSIZE - 1, 0)) != 0) {
        // recieve Error ó��
        if (recvLen == SOCKET_ERROR) {
            printf("Receive Error..\n");
            break;
        }

        // ���� �޽��� ���ڿ� ���� ���߱�
        message[recvLen] = '\0';
        printf("input msg from client : %s\n", message);
        retVal = SendMSG(dict, message, recvLen, clntCnt, clntSock, statusFlag, preMsg, r_net, weights);
        printf("retVal : %d\n", retVal);
        // ���� ��û �޽������� Ȯ��
        if (retVal == QUITREQ) {
            printf("Close Client Connection..\n");
            break;
        }
        statusFlag *= -1;

        // ���� �޽����� �ƴϸ� ���
        //printf("%s\n", message);
/*		for (int i = 0; i < BUFSIZE; i++) {
            message[i] = '\0';	// �ѹ� �������� ���ڿ� ����
        }*/
        Sleep(100);	// ���� ������ �ΰ� �ݺ�
    }

    SOCKET tmpAndroidClntSock = NULL;
    // ũ��Ƽ�� ����
    WaitForSingleObject(hMutex, INFINITE);
    {
        int boool = 0;
        char * tmpAndroidClntIp = NULL;

        for (i = 0; i < clntCnt; i++) {
            if (clntSock == clntSocks[i]) {
                for (int j = 0; j < 5; j++) {	// ����īƮ ip 5�� �߿���
                    if (strcmp(inet_ntoa(clntIpArr[i].sin_addr), scip[j]) == 0) {	// ������ �ִٸ�
                        boool = 1;	// ���� ���� ������ Ŭ���̾�Ʈ�� ����īƮ�̴�
                        // �� ����īƮ�� ¦�� ����Ʈ���� ip�� �̸� ��Ƶΰ�
                        tmpAndroidClntIp = dict_get(dict, inet_ntoa(clntIpArr[i].sin_addr));
                        // ���� ����īƮ�� ����Ʈ���̶� ¦�� ����
                        dict_remove(dict, inet_ntoa(clntIpArr[i].sin_addr));
                        break;
                    }
                }
            }
        }
        // ���� ������ Ŭ���̾�Ʈ�� Ŭ���̾�Ʈ �迭���� �����ϸ鼭 ���� Ŭ���̾�Ʈ ������ ������ ����.
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

        if (boool) {	// ���� ������ Ŭ���̾�Ʈ�� ����īƮ���� ��쿡�� ����Ʈ�� ���ӵ� ���� �����ش�
            // ����Ʈ�� ���ϵ� ã�Ƴ���
            for (i = 0; i < clntCnt; i++) {
                if (strcmp(tmpAndroidClntIp, inet_ntoa(clntIpArr[i].sin_addr)) == 0) {
                    tmpAndroidClntSock = clntSocks[i];
                    // ���� ������ Ŭ���̾�Ʈ�� Ŭ���̾�Ʈ �迭���� ����, ������ ����.
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
    // ũ��Ƽ�� ���� ��.

    // ������ Ŭ���̾�Ʈ ���� �ݱ�
    closesocket(clntSock);
    if (tmpAndroidClntSock != NULL) {
        closesocket(tmpAndroidClntSock);
    }
    return 0;
}


/* ��� Ŭ���̾�Ʈ�κ��� ������, � �޽��������� Ȯ���ϰ�
* Ŭ���̾�Ʈ���� �޽����� �����ϰų� object detection ���� Ư�� ������ �ϵ��� ����
* ���� : char *message		- ������ �޽���
*        int len			- �޽����� ����
*		 int clntCnt		- ���� ������� ��
*        unsigned clntSock - ��û�� ������� ���� ��ũ����
*/
int SendMSG(Dictionary *dict, char *message, int len, int clntCnt, unsigned clntSock, int flag, char *msg, network r_net, char *weights)
{
    char sendBuf[BUFSIZE];		// �۽� ����
    printf("%d flag:::::::::::", flag);

    int i, j;
    FILE *file_pointer;
    //file_pointer = fopen("C:\\Users\\JunSeop\\Desktop\\stat\\log.txt", "a+");   ////////////////////////////////////////////////////////////////// �ٸ� ȣ��Ʈ���� �� �� �̺κ� �� !�ٲٱ�
    file_pointer = fopen("D:\\log.txt", "a+");
    objcnt objects[5];
    for (int k = 0; k < 5; k++)
        objects[k].cnt = 0;
    strcpy(objects[0].name, "coca-cola");
    strcpy(objects[1].name, "butter_ring");
    strcpy(objects[2].name, "ramen");
    strcpy(objects[3].name, "downy");
    strcpy(objects[4].name, "corn-frost");
    
    // ũ��Ƽ�� ����
    WaitForSingleObject(hMutex, INFINITE);
    {
        // ���� ��û ��
        if (strcmp(message, "exit") == 0)
        {
            ReleaseMutex(hMutex);
            // ũ��Ƽ�� ���� ��.
            return QUITREQ;
        }

        int boool = 0;
        // ���� ����� ¦�� ���ٸ�
        for (i = 0; i < clntCnt; i++) {	// Ŭ���̾�Ʈ ip���� �� Ȯ���غ��ǵ�
            if (clntSock == clntSocks[i]) {	// �� Ŭ���̾�Ʈ�� ����
                printf("\n\ncurrent client num : %d\n\n\n", i);
                for (j = 0; j < 5; j++) {	// ����īƮ ip 5�� �߿���
                    if (strcmp(inet_ntoa(clntIpArr[i].sin_addr), scip[j]) == 0) {	// ������ �ִٸ�
                        boool = 1;	// �� �޽��� �۽����� ����īƮ�̴�.
                        break;
                    }
                }
                if (!(boool)) {	// �۽����� ����īƮ�� �ƴϰ� ����Ʈ�� ip�� ���
                    // �޸𸮻󿡼� �������·� ����īƮ�� ����Ʈ���� ¦ �����ش�.
                    // ����Ʈ�� Ŭ���̾�Ʈ������ ¦�������� ���� �ѹ� �޽����� �۽��� ����
                    int isIpRegex = 0;
                    // ���ӵ� Ŭ���̾�Ʈ���� ip �� �ϳ��� �����϶�� ��û���� �ƴ��� ���� �Ǻ�
                    //printf("%dsss\n", clntCnt);
                    for (j = 0; j < clntCnt; j++) {
                        //printf("A : %sl\nB : %sl\n%d***\n",message, inet_ntoa(clntIpArr[j].sin_addr), strcmp(message, inet_ntoa(clntIpArr[j].sin_addr)));
                        if (strcmp(message, inet_ntoa(clntIpArr[j].sin_addr)) == 0) {
                            isIpRegex = 1;
                        }
                    }
                    printf("Check %d AND %d\n", isIpRegex, dict_has(dict, message));
                    // ip ���ο�û�� ������, ���� ��ųʸ� ü�� ���� �� ����īƮ�� Ű�μ� ����Ǿ����� ������
                    if ((isIpRegex) && (!(dict_has(dict, message)))) {
                        printf("mapping\n\n\n");
                        dict_add(dict, message, inet_ntoa(clntIpArr[i].sin_addr));	// ��ųʸ� ü�� �߰�
                    }
                    // �׷��� ���� ���
                    else {
                        /*for (int i = 0; i < BUFSIZE; i++) {
                            message[i] = '\0';	// �ѹ� �������� ���ڿ� ����
                        }*/
                        strcpy(message, "bad request\n");
                        send(clntSock, message, strlen(message), 0);
                    }
                }
                else {	// �۽����� ����īƮ�� ���
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
                        // raspi�κ��� ���ŵ� ���ڿ��� ���� �������� �߶� ������� ���� argc(���� �ٲټ���)�� count, argv���ڿ� �迭 �����
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
                            
                            // �޽��� ����ش�
                            for (int x = 0; x < BUFSIZE; x++) {
                                msg[x] = 0;
                            }
                            // ���ο� �޽����� �־��ش�.
                            for (int z = 0; z < 5; z++)
                            {
                                sprintf(p1, "%d", objects[z].cnt);
                                sprintf(p2, "%d", z + 1);
                                fputs(objects[z].name, file_pointer);
                                fputs("��ü�� ", file_pointer);
                                fputs(p1, file_pointer);
                                fputs("�� �ֽ��ϴ�.\n", file_pointer);

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
                            // ¦ ����Ʈ�� ���� ã�Ƽ�
                            if (strcmp(inet_ntoa(clntIpArr[j].sin_addr), destClnt) == 0) {
                                break;
                            }
                        }
                        if (j >= 0 && j < clntCnt) {
                            if (strcmp(message, "=") != 0) {    // ���� ��ȭ�� �ִ� ��쿡��
                                strcat(msg, message);   // ���� �޽����� ���� �޽����� �߰��ϰ� (+ or -)
                                strcat(msg, ";\n");      // ������ �߰��ϰ�
                                strcpy(message, msg);	// test case
                                printf("sendmsg....: %sdddd", message);
                                send(clntSocks[j], message, strlen(message), 0);	// ¦ ����Ʈ������ �޼����� ������
                            }
                        }
                        
                    }
                }
            }
        }
    }
    ReleaseMutex(hMutex);
    // ũ��Ƽ�� ���� ��.
    return 0;
}


// ���� �ڵ鸵 �Լ�
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
    assert(dictionary->head->key != NULL);	//	key���� null�̸� �ȵȴ�
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