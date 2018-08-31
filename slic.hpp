#ifndef MY_SLIC
#define MY_SLIC
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

/*
parameter:
	int _SPixelNum : number of super pixels you expected to get.
	float _m: measure the importance of color and Coordinate distance.
	cv::InputArray _src: input image of type CV_8UC3 or CV_16UC3.
	cv::OutputArray _label: the labeled image.
	cv::OutputArray _centerLut: has 5 rows, rows 0-2 store the center lab color values,rows 3-4 store the center x and y coordinate. 
----------------------------------------------------------------------------------------
example:
	cv::Mat src = imread(".....");
	cv::Ptr<slic::SLIC> p = slic::createSLIC(int _SPixelNum,float _m);
	p->apply(cv::InputArray _src, cv::OutputArray _label, cv::OutputArray _centerLut);
	p->imshowLabel(string name);
*/
namespace slic {

	class SLIC {
	public:
		virtual void apply(cv::InputArray _src, cv::OutputArray _label, cv::OutputArray _centerLut) = 0;

		virtual void imshowLabel(std::string name) = 0;

		virtual void collectGarbage() = 0;
	};

	template<class T>
	class SLIC_calcu_mean_center_body :public cv::ParallelLoopBody {
	public:
		SLIC_calcu_mean_center_body(const cv::Mat& _labSrc, const cv::Mat& _centerLut, const cv::Mat& _labelMat, const int& _S) :
			labSrc(_labSrc), centerLut(_centerLut), labelMat(_labelMat), S(_S) {
		}

		void operator()(const cv::Range& range) const {
			int step = labSrc.cols / S;
			int* ptl = centerLut.ptr<int>(0);
			int* pta = centerLut.ptr<int>(1);
			int* ptb = centerLut.ptr<int>(2);
			int* ptx = centerLut.ptr<int>(3);
			int* pty = centerLut.ptr<int>(4);
			int rows = labSrc.rows;
			int cols = labSrc.cols;
			for (int i = range.start; i < range.end; i++) {
				ptl += i; pta += i; ptb += i; ptx += i; pty += i;
				int l = 0, a = 0, b = 0, x = 0, y = 0;
				int count = 0;

				for (int wy = std::max(0, (*pty) - S), wye = std::min(rows - 1, (*pty) + S); wy <= wye; wy++) {
					for (int wx = std::max(0, (*ptx) - S), wxe = std::min(cols - 1, (*ptx) + S); wx <= wxe; wx++) {
						cv::Vec<T, 3> v = labSrc.at<cv::Vec<T, 3>>(wy, wx);
						if (i == labelMat.at<int>(wy, wx)) {
							l += v[0]; a += v[1]; b += v[2];
							x += wx; y += wy;
							count++;
						}
					}
				}
				*ptl = int(l / count);
				*pta = int(a / count);
				*ptb = int(b / count);
				*ptx = int(x / count);
				*pty = int(y / count);
			}
		}

	private:
		cv::Mat labSrc;
		mutable cv::Mat centerLut;
		cv::Mat labelMat;
		int S;
	};

	template<class T>
	class SLIC_move_center_body :public cv::ParallelLoopBody {
	public:
		SLIC_move_center_body(const cv::Mat& _src, const cv::Mat& _labSrc, const cv::Mat& _centerLut, const cv::Matx33f _xkernel, const cv::Matx33f _ykernel, const int& _S) :
			src(_src), labSrc(_labSrc), centerLut(_centerLut), xkernel(_xkernel), ykernel(_ykernel), S(_S) {

		}

		void operator()(const cv::Range& range) const {
			int step = src.cols / S;
			int* ptl = centerLut.ptr<int>(0);
			int* pta = centerLut.ptr<int>(1);
			int* ptb = centerLut.ptr<int>(2);
			int* ptx = centerLut.ptr<int>(3);
			int* pty = centerLut.ptr<int>(4);

			for (int i = range.start; i < range.end; i++) {
				int y = S / 2 + (i / step)*S;
				int x = S / 2 + (i % step)*S;
				ptl += i; pta += i; ptb += i; ptx += i; pty += i;

				float min_grad = INFINITY;
				for (int wy = y - 1; wy <= y + 1; wy++)
					for (int wx = x - 1; wx <= x + 1; wx++) {

						//computer gradient
						float gradx = 0, grady = 0, grad = 0;
						for (int gwy = wy - 1; gwy <= wy + 1; gwy++)
							for (int gwx = wx - 1; gwx <= wx + 1; gwx++) {
								cv::Vec<T, 3> pixel = src.at<cv::Vec<T, 3>>(gwy, gwx);
								float gv = (0.299f*(pixel[2])) + (0.587f*(pixel[1])) + (0.114f*(pixel[0]));//BGR2Gray
								gradx += gv * xkernel(gwy - wy + 1, gwx - wx + 1);
								grady += gv * xkernel(gwy - wy + 1, gwx - wx + 1);
							}
						grad = std::sqrt(gradx*gradx + grady * grady);

						if (min_grad > grad) {
							cv::Vec<T, 3> labv = labSrc.at<Vec<T, 3>>(wy, wx);
							*ptl = labv[0];
							*pta = labv[1];
							*ptb = labv[2];
							*ptx = wx;
							*pty = wy;
						}
					}

			}
		}

	private:
		cv::Mat src;
		cv::Mat labSrc;
		mutable cv::Mat centerLut;
		cv::Matx33f xkernel;
		cv::Matx33f ykernel;
		int S;
	};

	template <class T>
	int SLIC_initialize_centers(const cv::Mat& _src, cv::Mat& _labSrc, cv::Mat& _centerLut, const int& _SPixelNum)
	{
		const cv::Matx33f xkernel = { -3.f,0.f,3.f,-10.f,0.f,10.f,-3.f,0.f,3.f };
		const cv::Matx33f ykernal = { -3.f,-10.f,-3.f,0.f,0.f,0.f,3.f,10.f,3.f };

		int rows = _src.rows;
		int cols = _src.cols;
		const int N = rows * cols;
		const int S = (int)std::sqrt(N / _SPixelNum); CV_Assert(S > 3);

		const int SPixelNum = (rows / S)*(cols / S);
		_centerLut.create(5, SPixelNum, CV_32SC1);

		cv::cvtColor(_src, _labSrc, CV_BGR2Lab);
		cv::Ptr<cv::ParallelLoopBody> moveCenterBody = cv::makePtr<SLIC_move_center_body<T>>(_src, _labSrc, _centerLut, xkernel, ykernal, S);
		cv::parallel_for_(cv::Range(0, SPixelNum), *moveCenterBody);
		return S;
	}

	template<class T>
	void SLIC_calcu_loop(const cv::Mat& _labSrc, cv::Mat& _centerLut, cv::Mat& _labelMat, const int& _S, const float& _m)
	{
		typedef cv::Vec<T, 3> MVEC;
		int rows = _labSrc.rows;
		int cols = _labSrc.cols;
		_labelMat.create(_labSrc.size(), CV_32SC1);
		//_labelMat.setTo(-1);

		const int SPixelsNum = _centerLut.cols;
		int c = 10;//Ñ­»·´ÎÊý
		while (c--) {
			cv::Mat minDist(_labelMat.size(), CV_32FC1, INFINITY);

			int* ptl = _centerLut.ptr<int>(0);
			int* pta = _centerLut.ptr<int>(1);
			int* ptb = _centerLut.ptr<int>(2);
			int* ptx = _centerLut.ptr<int>(3);
			int* pty = _centerLut.ptr<int>(4);
			for (int k = 0; k < SPixelsNum; k++, ptl++, pta++, ptb++, ptx++, pty++) {

				int x = *ptx;
				int y = *pty;
				int l = *ptl;
				int a = *pta;
				int b = *ptb;

				for (int wy = std::max(0, y - _S), wye = std::min(rows - 1, y + _S); wy <= wye; wy++) {

					for (int wx = std::max(0, x - _S), wxe = std::min(cols - 1, x + _S); wx <= wxe; wx++) {

						MVEC v = _labSrc.at<MVEC>(wy, wx);
						int dc = (l - v[0])*(l - v[0]) + (a - v[1])*(a - v[1]) + (b - v[2])*(b - v[2]);
						int ds = (x - wx)*(x - wx) + (y - wy)*(y - wy);

						float d = std::sqrt(dc + ds * _m*_m / _S / _S);
						if (d < minDist.at<float>(wy, wx)) {
							minDist.at<float>(wy, wx) = d;
							_labelMat.at<int>(wy, wx) = k;
						}
					}
				}

			}
			cv::Ptr<cv::ParallelLoopBody> calcuMeanCenterBody = cv::makePtr<SLIC_calcu_mean_center_body<T>>(_labSrc, _centerLut, _labelMat, _S);
			cv::parallel_for_(cv::Range(0, SPixelsNum), *calcuMeanCenterBody);
		}
	}
	
	class SLIC_Impl :public slic::SLIC {
	public:
		SLIC_Impl(int _SPixelNum = 300, float _m = 25.f) :SPixelNum(_SPixelNum), m(_m) {

		}

		void apply(cv::InputArray _src, cv::OutputArray _label, cv::OutputArray _centerLut);

		void collectGarbage();

		void imshowLabel(std::string name);
	private:
		cv::Mat src;
		cv::Mat label;
		cv::Mat centerLut;
		int SPixelNum;
		float m;
	};

	void SLIC_Impl::collectGarbage()
	{
		src.release();
		label.release();
		centerLut.release();
	}

	void SLIC_Impl::apply(cv::InputArray _src, cv::OutputArray _label, cv::OutputArray _centerLut)
	{
		CV_Assert(_src.type() == CV_8UC3 || _src.type() == CV_16UC3);

		src = _src.getMat();
		label = _label.getMat();
		centerLut = _centerLut.getMat();

		cv::Mat labSrc;
		int S;
		if (_src.type() == CV_8UC3) {
			S = SLIC_initialize_centers<uchar>(src, labSrc, centerLut, SPixelNum);
			SLIC_calcu_loop<uchar>(labSrc, centerLut, label, S, m);
		}
		else {
			S = SLIC_initialize_centers<short>(src, labSrc, centerLut, SPixelNum);
			SLIC_calcu_loop<short>(labSrc, centerLut, label, S, m);
		}

		//collectGarbage();
	}

	void SLIC_Impl::imshowLabel(std::string name)
	{
		cv::RNG rng;
		cv::Mat showImg(label.size(), CV_8UC3);

		std::vector<cv::Vec3b> lut;
		for (int i = 0; i < SPixelNum; i++)
			lut.push_back(cv::Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));

		cv::MatIterator_<cv::Vec3b> ps = showImg.begin<cv::Vec3b>();
		cv::MatConstIterator_<int> pl = label.begin<int>();
		for (; ps != showImg.end<cv::Vec3b>(); ps++, pl++) {
			*ps = lut[*pl];
		}

		cv::imshow(name, showImg);
	}

	cv::Ptr<SLIC> createSLIC(int _SPixelNum = 300, float _m = 25.f)
	{
		return cv::makePtr<SLIC_Impl>(_SPixelNum, _m);
	}
}
#endif // !MY_SLIC

