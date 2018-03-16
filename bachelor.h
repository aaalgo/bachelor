#ifndef AAALGO_BACHELOR
#define AAALGO_BACHELOR

#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/core/noncopyable.hpp>

namespace bachelor {

    using std::vector;

    enum Order {
        NHWC = 1,
        NCHW = 2
    };

    // this is just a view
    // derived class has to setup buffer
    // so data can be copied.
    // derivied class must also ensure that buffer is set to 0 before destruction
    // this is a safety measure
    class BatchBase: boost::noncopyable {
    public:
        struct Config {
            int batch;
            int height;
            int width;
            int channels;
            int depth;           // as in opencv
            Order order;
            Config (): batch(0), height(0), width(0), channels(3), depth(CV_32F), order(NHWC) {
            }
            int channel_size () const {
                return height * width * CV_ELEM_SIZE1(depth);
            }
            int image_size () const {
                return channel_size() * channels;
            }
            int batch_size () const {
                return image_size() * batch;
            }
        };
        BatchBase (Config const &c): conf(c), channel_size(conf.channel_size()), image_size(conf.image_size()), batch_size(conf.batch_size()), buffer(nullptr), end(nullptr), next(nullptr) {
            reset();
        }
        ~BatchBase () {
            if (buffer) throw 0;
        }
        void reset () {
            if (!buffer) throw 0;
            next = reinterpret_cast<char *>(buffer);
            cnt = 0;
        }
        void skip_next () {
            if (!next) throw 0;
            if (next >= end) throw 0;
            next += image_size;
            ++cnt;
        }
        void fill_next_transform (cv::Mat const &image) {
            if (!image.data) {
                skip_next();
                return;
            }
            cv::Mat tmp = image;
            // convert color space
            if ((tmp.rows != conf.height)
                    || (tmp.cols != conf.width)) {
                cv::resize(tmp, stage1, cv::Size(conf.width, conf.height));
                tmp = stage1;
            }
            if (tmp.channels() != conf.channels) {
                if (tmp.channels() == 3 && conf.channels == 1) {
                    cv::cvtColor(tmp, stage2, CV_BGR2GRAY);
                }
                else if (tmp.channels() == 4 && conf.channels == 1) {
                    cv::cvtColor(tmp, stage2, CV_BGRA2GRAY);
                }
                else if (tmp.channels() == 4 && conf.channels == 3) {
                    cv::cvtColor(tmp, stage2, CV_BGRA2BGR);
                }
                else if (tmp.channels() == 1 && conf.channels == 3) {
                    cv::cvtColor(tmp, stage2, CV_GRAY2BGR);
                }
                else {
                    throw 0;
                }
                tmp = stage2;
            }
            if (tmp.depth() != conf.depth) {
                tmp.convertTo(stage3, conf.depth);
                tmp = stage3;
            }
            fill_next(tmp);
        }
        void fill_next (cv::Mat const &image) {
            if (!image.data) {
                skip_next();
                return;
            }
            if ((image.cols != conf.width)
                || (image.rows != conf.height)
                || (image.channels() != conf.channels)
                || (image.depth() != conf.depth)) {
                throw 0;
            }
            if (!next) throw 0;
            if (next >= end) throw 0;
            if (image.isContinuous()) throw 0;
            if (conf.order == NHWC || conf.channels == 1) {
                memcpy(next, image.ptr<char const>(0), image_size);
                next += image_size;
            }
            else {
                // copy by channel
                vector<cv::Mat> channels;
                cv::split(image, channels);
                for (unsigned i = 0; i < channels.size(); ++i) {
                    memcpy(next, channels[i].ptr<char const>(0), channel_size);
                    next += channel_size;
                }
            }
            ++cnt;
        }
        void fill_0 () {
            if (!next) throw 0;
            if (next >= end) return;
            bzero(next, end-next);
        }
    protected:
        Config conf;
        int channel_size;
        int image_size;
        int batch_size;
        int cnt;
        void *buffer;
        char *end;
        char *next;
        cv::Mat stage1;
        cv::Mat stage2;
        cv::Mat stage3;
    };

    class Batch: public BatchBase {
        vector<char> data;
    public:
        Batch (Config const &c): Batch(c), data(c.batch_size()) {
            buffer = &data[0];
        }
        Batch (Config const &c, void *data): Batch(c) {
            buffer = data;
        }
        ~Batch () {
            buffer = nullptr;
        }
        template <typename T = char>
        T *ptr () {
            return reinterpret_cast<T *>(buffer);
        }
    };

#ifdef NPY_NDARRAYOBJECT_H
    class NumpyBatch: public BatchBase {
        PyObject *batch;
        npy_intp dims[4];

        int np_type (int depth) {
            switch (depth) {
                case CV_8U: return NPY_UINT8;
                case CV_32F: return NPY_FLOAT32;
            }
            throw 0;
            return 0;
        }
    public:
        NumpyBatch (Config const &c): BatchBase(c) {
            if (c.order == NHWC) {
                dims[0] = conf.batch;
                dims[1] = conf.height;
                dims[2] = conf.width;
                dims[3] = conf.channels;
            }
            else {
                dims[0] = conf.batch;
                dims[1] = conf.channels;
                dims[2] = conf.height;
                dims[3] = conf.width;
            }
            batch = PyArray_SimpleNew(4, dims, np_type(conf.depth));
            if (!batch) throw 0;
            buffer = PyArray_DATA(batch);
        }

        ~NumpyBatch () {
            if (batch) throw 0;
        }

        PyObject *detach () {
            if (cnt > conf.batch) throw 0;
            if (cnt < conf.batch) {
                dims[0] = cnt;
                batch = PyArray_Resize((PyArrayObject *)batch, dims, 1, NPY_CORDER);
            }
            PyObject *r = batch;
            batch = nullptr;
            buffer = nullptr;
            return r;
        }
    };
#endif
}

#endif
