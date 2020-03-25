Image Processing Living Intelligence_Hardware
------
This is a repository for Server module of Image Processing Living Intelligence.<br/>

This project requires a **subnet** configuration between the server, the hardware and the Android module.

Requirements
------
1. Visual Studio 2017 이하 버전 설치<br/>
이따 설치할 CUDA에 사용될 CMake도 함께 설치해줘야합니다.<br/>
먼저 Visual Studio 설치가 완료된 분이시라면 visual studio installer를 이용하여 수정을 눌러주세요.<br/>

패키지 목록들 중 **데스크톱용  VC++ 2015.3 v14.00(v140) 도구 집합**이 설치되어있어야 합니다.<br/>
CUDA 설치 전에 미리 설치가 되어있어야합니다.<br/>

2. CUDA 설치<br/>
3. cuDNN 설치<br/>
[https://ohjinjin.github.io/machinelearning/darkflow-1/](https://ohjinjin.github.io/machinelearning/darkflow-1/)
<br/>위 링크에서 CUDA와 cuDNN 설치 과정만 따라해주시면 됩니다.<br/>
cuDNN의 **환경변수** 등록까지 마쳐주시기 바랍니다.<br/>

4. openCV 3.4.0 설치
openCV 3.4.0 다운로드 url : [https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.0/](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.0/)

추후 Visual Studio에서 OpenCV 라이브러리를 사용하기 위하여 시스템 환경변수에 추가해줍니다.<br/>
**내PC \> 마우스 우클릭 \> 속성 \>고급 시스템 설정 \> 환경 변수** 까지 진입하시면 팝업창이 확인되실 겁니다.<br/>

시스템 변수 중 path 변수에 값을 추가해줄 것이므로 path를 선택한 후 **편집** 버튼을 눌러서 openCV DLL파일이 설치된 경로를 적어주셔야합니다.<br/>

5. 이 프로젝트 다운로드 or 클론<br/>
집 해제 후 ..\\build\\darknet\\darknet.sln을 실행해줍니다. 그러면 vs studio 2017을 통하여 프로젝트가 열릴 텐데 상단 메뉴에서 Release, x64로 변경해줍니다.<br/>

openCV 라이브러리를 올바르게 가져다 사용할 수 있도록 프로젝트 속성을 변경해줍니다.<br/><br/>
**프로젝트 속성 \> C/C\+\+ \> 일반 \> 추가 포함 디렉터리**에서 opencv 경로를 넣어줍니다.<br/>

<br/>**프로젝트 속성 \> 링커 \> 일반 \> 추가 라이브러리 디렉터리**에서 opencv 라이브러리 경로를 넣어줍니다.<br/>

빌드 종속성을 자신의 CUDA 버전에 맞게 수정합니다.<br/>
<br/>CUDA 10.0을 선택하고 확인버튼을 눌러주세요.<br/>

여기까지 완료했다면 ctrl + shift + b를 눌러 프로젝트를 빌드해주면됩니다.<br/>

6. 하드웨어 모듈의 IP부분, 즉 쇼핑카트 IP 부분을 수정해주세요.<br/><br/>

Sequence of operation of the whole system
------
모듈별 요구사항 및 빌드 준비가 완료되었다면 서버를 가장 먼저 실행시켜주세요.<br/>

실행 시킬때에는 port 번호를 8080으로 지정해주면서 실행시켜야합니다.<br/>

이후 쇼핑카트 하드웨어 모듈을 실행시켜주세요.<br/>
그러면 서버와 쇼핑카트 간의 TCP/IP 서버&클라이언트 연결이 완료될 겁니다.<br/>
기 충전중이던 쇼핑카트를 사용자가 사용하기 시작하면서 방전이 시작되고 컴퓨팅 파워가 공급됨과 동시에 하드웨어 모듈에서는 내장 프로그램이 자동 실행됩니다.<br/>

그 다음으로 사용자의 스마트디바이스에서 안드로이드 어플리케이션을 실행시키고 로그인을 해주신 뒤 가상장바구니 UI로 진입해주세요.<br/>

이 때 최초 한 번 쇼핑카트의 IP를 입력하도록 구현하였는데, 쇼핑카트 IP를 올바르게 입력했다면 서버와 안드로이드 모듈간의 TCP/IP 서버&클라이언트 연결이 완료되며, 해당 쇼핑카트와 스마트 디바이스간 1:1 매핑이 서버상에서 이루어지게 됩니다.<br/>

사용자는 비로소 쇼핑을 시작하면 됩니다.<br/>
쇼핑카트에 상품을 투입하면 하드웨어 모듈에서 트리거 감지를 통해 실시간 영상에 대해 웹스트리밍을 제공하며 서버가 이를 전달받아 딥러닝 기술인 객체 탐지를 진행하여 어떤 상품인지, 그 수량은 각각 어떻게 되는지를 파악합니다.<br/>
이후의 무게 증감변화를 하드웨어 모듈에서 파악하여 상품이 들어간 것인지 나간 것인지에 대한 판단 여부까지 포함해 이 데이터들을 직렬화하여 안드로이드 모듈에 전송합니다.<br/>

안드로이드 어플리케이션의 가상 장바구니 UI에서는 이 내용이 반영되며, 쇼핑이 완료되면 자가 결제를 하고 쇼핑카트를 반납함으로써 본 시스템 사용 시나리오가 종료됩니다.<br/>

Usage
------
시연 영상 : [https://youtu.be/Cik78MP8W3E](https://youtu.be/Cik78MP8W3E)
<br/>
관련 포스팅 : [https://ohjinjin.github.io/projects/IPLI/](https://ohjinjin.github.io/projects/IPLI/)
<br/>
KIC 사이트 : [https://statkclee.github.io/kic2020/#pr](https://statkclee.github.io/kic2020/#pr)
