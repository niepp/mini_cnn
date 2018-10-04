#ifndef __NETWORK_H__
#define __NETWORK_H__

#include <cassert>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>

namespace mini_cnn
{
class network
{
public:
	input_layer *m_input_layer;
	output_layer *m_output_layer;
	std::vector<layer_base*> m_layers;

public:
	network()
	{
	}

	void add_layer(layer_base *layer)
	{
		if (m_output_layer != nullptr)
		{
			throw std::exception("add layer after output!");
		}

		if (m_input_layer == nullptr)
		{
			input_layer *in = dynamic_cast<input_layer*>(layer);
			if (in == nullptr)
			{
				throw std::exception("must add input layer first!");
			}
			m_layers.push_back(layer);
			m_input_layer = in;
		}
		else
		{
			layer_base* last = *m_layers.rbegin();
			last->connect(layer);
			m_layers.push_back(layer);

			output_layer *out = dynamic_cast<output_layer*>(layer);
			if (out != nullptr)
			{
				m_output_layer = out;
				m_output_layer->connect(nullptr);
			}
		}
	}

	void init_all_weight(normal_random nrand)
	{
		//1. Gaussian initialize
		//	Weights are randomly drawn from Gaussian distributions with fixed mean(e.g., 0) and fixed standard deviation(e.g., 0.01).
		//	This is the most common initialization method in deep learning.
		//2. Xavier initialize

		//3. He initialize
		//  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.Delving Deep into Rectifiers : Surpassing Human - Level Performance on ImageNet Classification, Technical report, arXiv, Feb. 2015

		int_t len = static_cast<int_t>(m_layers.size());
		for (int_t i = 0; i < len; ++i)
		{
			layer_base *layer = m_layers[i];
			layer->init_weight(nrand);
		}
	}

	void set_task_count(int_t task_count)
	{
		int_t len = static_cast<int_t>(m_layers.size());
		for (int_t i = 0; i < len; ++i)
		{
			layer_base *layer = m_layers[i];
			layer->set_task_count(task_count);
		}
	}

	void train(const varray_vec &batch_img_vec, const varray_vec &batch_label_vec, float_t eta, const int_t max_threads)
	{
		nn_assert(batch_img_vec.size() == batch_label_vec.size());
		int_t batch_size = batch_img_vec.size();
		int_t nthreads = std::min(max_threads, batch_size);
		int_t nstep = (batch_size + nthreads - 1) / nthreads;

		pre_train();

		std::vector<std::future<void>> futures;
		for (int_t k = 0; k < nthreads && k * nstep < batch_size; ++k)
		{
			int_t begin = k * nstep;
			int_t end = std::min(batch_size, begin + nstep);
			futures.push_back(std::move(std::async(std::launch::async, [&, begin, end, k]() {
				train_task(batch_img_vec, batch_label_vec, begin, end, k);
			})));
		}
		for (auto &future : futures)
		{
			future.wait();
		}
		float_t eff = eta / batch_size;
		update_all_weight(eff);
	}

	int_t test(const varray_vec &test_img_vec, const index_vec &test_lab_vec, const int_t max_threads)
	{
		nn_assert(test_img_vec.size() == test_lab_vec.size());
		int_t test_count = test_img_vec.size();

		int_t nthreads = max_threads;
		int_t nstep = (test_count + nthreads - 1) / nthreads;

		std::vector<std::future<int_t>> futures;
		for (int_t k = 0; k < nthreads && k * nstep < test_count; ++k)
		{
			int_t begin = k * nstep;
			int_t end = std::min(test_count, begin + nstep);
			futures.push_back(std::move(std::async(std::launch::async, [&, begin, end, k]() {
				return test_task(test_img_vec, test_lab_vec, begin, end, k);
			})));
		}
		int_t correct = 0;
		for (auto &future : futures)
		{
			correct += future.get();
		}
		return correct;
	}

	float_t get_cost(const varray_vec &img_vec, const varray_vec &lab_vec, const int max_threads)
	{
		nn_assert(img_vec.size() == lab_vec.size());
		int tot_count = img_vec.size();

		int nthreads = max_threads;
		int nstep = (tot_count + nthreads - 1) / nthreads;

		std::vector<std::future<float_t>> futures;
		for (int k = 0; k < nthreads && k * nstep < tot_count; ++k)
		{
			int begin = k * nstep;
			int end = std::min(tot_count, begin + nstep);
			futures.push_back(std::move(std::async(std::launch::async, [&, begin, end, k]() {
				return cost_task(img_vec, lab_vec, begin, end, k);
			})));
		}
		float_t tot_cost = 0;
		for (auto &future : futures)
		{
			tot_cost += future.get();
		}
		if (tot_count > 0)
		{
			tot_cost /= tot_count;
		}
		return tot_cost;
	}

	bool gradient_check(const varray &test_img, const varray &test_lab, float_t precision = 1e-4)
	{
		nn_assert(!m_layers.empty());

		set_task_count(1);

		pre_train();

		bool check_ok = true;
		int_t len = static_cast<int_t>(m_layers.size());
		for (auto &layer : m_layers)
		{
			auto &ts = layer->get_task_storage(0);
			varray &w = layer->m_w;
			varray &dw = ts.m_dw;
			int_t w_sz = w.size();
			for (int_t i = 0; i < w_sz; ++i)
			{
				if (!calc_gradient(test_img, test_lab, w[i], dw[i], precision))
				{
					check_ok = false;
				}
			}

			varray &b = layer->m_b;
			varray &db = ts.m_db;
			int_t b_sz = b.size();
			for (int_t i = 0; i < b_sz; ++i)
			{
				if (!calc_gradient(test_img, test_lab, b[i], db[i], precision))
				{
					check_ok = false;
				}
			}

		}
		return check_ok;
	}

private:
	void pre_train()
	{
		int_t len = static_cast<int_t>(m_layers.size());
		for (int_t i = 0; i < len; ++i)
		{
			layer_base *layer = m_layers[i];
			layer->pre_train();
		}
	}

	const varray& forward(const varray& input, int_t task_idx)
	{
		return m_input_layer->forw_prop(input, task_idx);
	}

	void backward(const varray &label, int_t task_idx)
	{
		m_output_layer->backward(label, task_idx);
	}

	void update_all_weight(float_t eff)
	{
		for (auto &layer : m_layers)
		{
			layer->update_weights(eff);
		}
	}

	void train_task(const varray_vec &batch_img_vec, const varray_vec &batch_label_vec, int_t begin, int_t end, int_t task_idx)
	{
		for (int_t i = begin; i < end; ++i)
		{
			forward(*batch_img_vec[i], task_idx);
			backward(*batch_label_vec[i], task_idx);
		}
	}

	int_t test_task(const varray_vec &test_img_vec, const index_vec &test_lab_vec, int_t begin, int_t end, int_t task_idx)
	{
		int_t c_count = 0;
		for (int_t i = begin; i < end; ++i)
		{
			const varray&out = forward(*test_img_vec[i], task_idx);
			int_t lab = out.arg_max();
			if (lab == test_lab_vec[i])
			{
				++c_count;
			}
		}
		return c_count;
	}

	float_t cost_task(const varray_vec &img_vec, const varray_vec &label_vec, int_t begin, int_t end, int_t task_idx)
	{
		float_t cost = 0;
		for (int_t i = begin; i < end; ++i)
		{
			const varray&out = m_input_layer->forw_prop(*img_vec[i], task_idx);
			cost += m_output_layer->calc_cost(false, *label_vec[i], task_idx);
		}
		return cost;
	}

	bool calc_gradient(const varray &test_img, const varray &test_lab, float_t &w, float_t &dw, float_t precision)
	{
		static const float_t EPSILON = 1e-6;

		float_t prev_w = w;
		w = prev_w + EPSILON;
		m_input_layer->forw_prop(test_img, 0);
		float_t loss_0 = m_output_layer->calc_cost(true, test_lab, 0);

		w = prev_w - EPSILON;
		m_input_layer->forw_prop(test_img, 0);
		float_t loss_1 = m_output_layer->calc_cost(true, test_lab, 0);
		float_t delta_by_numerical = (loss_0 - loss_1) / (float_t(2.0) * EPSILON);

		w = prev_w;
		this->pre_train();
		m_input_layer->forw_prop(test_img, 0);
		m_output_layer->backward(test_lab, 0);

		float_t delta_by_bprop = dw;

		if (!f_is_valid(loss_0) || !f_is_valid(loss_1) || !f_is_valid(dw))
		{
			std::cout << "[overflow] loss_0:" << loss_0 << "\tloss_1:" << loss_1 << "\tdw:" << dw << std::endl;
			return false;
		}

		float_t absError = std::abs(delta_by_bprop - delta_by_numerical);
		bool correct = absError <= precision;
		if (!correct)
		{
			std::cout << "bprop:" << delta_by_bprop << "\tnumerical:" << delta_by_numerical << std::endl;
		}
		return correct;
	}

};

}

#endif //__NETWORK_H__

