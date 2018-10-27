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
private:
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

	nn_int paramters_count() const
	{
		nn_int cnt = 0;
		for (auto &layer : m_layers)
		{
			cnt += layer->paramters_count();
		}
		return cnt;
	}

	void init_all_weight(weight_initializer &initializer)
	{
		initializer(m_layers);
	}

	void set_task_count(nn_int task_count)
	{
		for (auto &layer : m_layers)
		{
			layer->set_task_count(task_count);
		}
	}

	nn_float SGD(const varray_vec &img_vec, const varray_vec &lab_vec, const varray_vec &test_img_vec, const index_vec &test_lab_vec
		, std::mt19937_64 generator, nn_int epoch, nn_int batch_size, nn_float learning_rate, nn_int nthreads
		, std::function<void(nn_int, nn_int)> minibatch_callback
		, std::function<void(nn_int, nn_int, nn_float, nn_float, nn_float)> epoch_callback)
	{
		set_task_count(nthreads);

		nn_float max_accuracy = 0;
		nn_int img_count = img_vec.size();
		nn_int test_img_count = test_img_vec.size();
		nn_int batch = img_count / batch_size;

		std::vector<nn_int> idx_vec(img_count);
		for (nn_int k = 0; k < img_count; ++k)
		{
			idx_vec[k] = k;
		}

		for (nn_int c = 0; c < epoch; ++c)
		{
			nn_float tstart = get_now_ms();
			std::shuffle(idx_vec.begin(), idx_vec.end(), generator);
			varray_vec batch_img_vec(batch_size);
			varray_vec batch_label_vec(batch_size);
			for (nn_int i = 0; i < batch; ++i)
			{
				for (nn_int k = 0; k < batch_size; ++k)
				{
					nn_int j = idx_vec[(i * batch_size + k) % img_count];
					batch_img_vec[k] = img_vec[j];
					batch_label_vec[k] = lab_vec[j];
				}
				train_one_batch(batch_img_vec, batch_label_vec, learning_rate, nthreads);
				minibatch_callback((i + 1) * batch_size, img_count);
				break;
			}
			break;
			nn_float tend = get_now_ms();
			nn_float elapse = (tend - tstart) * 0.001f;
			nn_int correct = test(test_img_vec, test_lab_vec, nthreads);
			nn_float cur_accuracy = (1.0f * correct / test_img_count);
			max_accuracy = std::max(max_accuracy, cur_accuracy);
			nn_float tot_cost = get_cost(img_vec, lab_vec, nthreads);
			epoch_callback(c + 1, epoch, cur_accuracy, tot_cost, elapse);
		}
		return max_accuracy;
	}

	void train_one_batch(const varray_vec &batch_img_vec, const varray_vec &batch_label_vec, nn_float eta, const nn_int max_threads)
	{
		nn_assert(batch_img_vec.size() == batch_label_vec.size());
		nn_int batch_size = batch_img_vec.size();
		nn_int nthreads = std::min(max_threads, batch_size);
		nn_int nstep = (batch_size + nthreads - 1) / nthreads;

		std::vector<std::future<void>> futures;
		for (nn_int k = 0; k < nthreads && k * nstep < batch_size; ++k)
		{
			nn_int begin = k * nstep;
			nn_int end = std::min(batch_size, begin + nstep);
			futures.push_back(std::move(std::async(std::launch::async, [&, begin, end, k]() {
				train_task(batch_img_vec, batch_label_vec, begin, end, k);
			})));
		}
		for (auto &future : futures)
		{
			future.wait();
		}
		nn_float eff = eta / batch_size;
		update_all_weight(eff);
	}

	nn_int test(const varray_vec &test_img_vec, const index_vec &test_lab_vec, const nn_int max_threads)
	{
		nn_assert(test_img_vec.size() == test_lab_vec.size());
		nn_int test_count = test_img_vec.size();

		nn_int nthreads = max_threads;
		nn_int nstep = (test_count + nthreads - 1) / nthreads;

		std::vector<std::future<nn_int>> futures;
		for (nn_int k = 0; k < nthreads && k * nstep < test_count; ++k)
		{
			nn_int begin = k * nstep;
			nn_int end = std::min(test_count, begin + nstep);
			futures.push_back(std::move(std::async(std::launch::async, [&, begin, end, k]() {
				return test_task(test_img_vec, test_lab_vec, begin, end, k);
			})));
		}
		nn_int correct = 0;
		for (auto &future : futures)
		{
			correct += future.get();
		}
		return correct;
	}

	nn_float get_cost(const varray_vec &img_vec, const varray_vec &lab_vec, const nn_int max_threads)
	{
		nn_assert(img_vec.size() == lab_vec.size());
		nn_int tot_count = img_vec.size();

		nn_int nthreads = max_threads;
		nn_int nstep = (tot_count + nthreads - 1) / nthreads;

		std::vector<std::future<nn_float>> futures;
		for (nn_int k = 0; k < nthreads && k * nstep < tot_count; ++k)
		{
			nn_int begin = k * nstep;
			nn_int end = std::min(tot_count, begin + nstep);
			futures.push_back(std::move(std::async(std::launch::async, [&, begin, end, k]() {
				return cost_task(img_vec, lab_vec, begin, end, k);
			})));
		}
		nn_float tot_cost = 0;
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

	bool gradient_check(const varray &test_img, const varray &test_lab, nn_float precision = 1e-4)
	{
		nn_assert(!m_layers.empty());

		set_task_count(1);

		bool check_ok = true;
		for (auto &layer : m_layers)
		{
			auto &ts = layer->get_task_storage(0);
			varray &w = layer->m_w;
			varray &dw = ts.m_dw;
			nn_int w_sz = w.size();
			for (nn_int i = 0; i < w_sz; ++i)
			{
				if (!calc_gradient(test_img, test_lab, w[i], dw[i], precision))
				{
					check_ok = false;
				}
			}

			varray &b = layer->m_b;
			varray &db = ts.m_db;
			nn_int b_sz = b.size();
			for (nn_int i = 0; i < b_sz; ++i)
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
	void clear_all_grident()
	{
		for (auto &layer : m_layers)
		{
			layer->clear_grident();
		}
	}

	void forward(const varray& input, nn_int task_idx)
	{
		m_input_layer->forw_prop(input, task_idx);
	}

	void backward(const varray &label, nn_int task_idx)
	{
		m_output_layer->backward(label, task_idx);
	}

	void update_all_weight(nn_float eff)
	{
		for (auto &layer : m_layers)
		{
			layer->update_weights(eff);
		}
	}

	void train_task(const varray_vec &batch_img_vec, const varray_vec &batch_label_vec, nn_int begin, nn_int end, nn_int task_idx)
	{
		for (nn_int i = begin; i < end; ++i)
		{
			forward(*batch_img_vec[i], task_idx);
			backward(*batch_label_vec[i], task_idx);
		}
	}

	nn_int test_task(const varray_vec &test_img_vec, const index_vec &test_lab_vec, nn_int begin, nn_int end, nn_int task_idx)
	{
		nn_int c_count = 0;
		for (nn_int i = begin; i < end; ++i)
		{
			forward(*test_img_vec[i], task_idx);
			nn_int lab = m_output_layer->get_output(task_idx).arg_max();
			if (lab == test_lab_vec[i])
			{
				++c_count;
			}
		}
		return c_count;
	}

	nn_float cost_task(const varray_vec &img_vec, const varray_vec &label_vec, nn_int begin, nn_int end, nn_int task_idx)
	{
		nn_float cost = 0;
		for (nn_int i = begin; i < end; ++i)
		{
			m_input_layer->forw_prop(*img_vec[i], task_idx);
			cost += m_output_layer->calc_cost(false, *label_vec[i], task_idx);
		}
		return cost;
	}

	bool calc_gradient(const varray &test_img, const varray &test_lab, nn_float &w, nn_float &dw, nn_float precision)
	{
		static const nn_float EPSILON = 1e-6f;

		clear_all_grident();

		nn_float prev_w = w;
		w = prev_w + EPSILON;
		m_input_layer->forw_prop(test_img, 0);
		nn_float loss_0 = m_output_layer->calc_cost(true, test_lab, 0);

		w = prev_w - EPSILON;
		m_input_layer->forw_prop(test_img, 0);
		nn_float loss_1 = m_output_layer->calc_cost(true, test_lab, 0);
		nn_float delta_by_numerical = (loss_0 - loss_1) / (nn_float(2.0) * EPSILON);

		w = prev_w;
		m_input_layer->forw_prop(test_img, 0);
		m_output_layer->backward(test_lab, 0);

		nn_float delta_by_bprop = dw;

		if (!f_is_valid(loss_0) || !f_is_valid(loss_1) || !f_is_valid(dw))
		{
			std::cout << "[overflow] loss_0:" << loss_0 << "\tloss_1:" << loss_1 << "\tdw:" << dw << std::endl;
			return false;
		}

		nn_float absError = std::abs(delta_by_bprop - delta_by_numerical);
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

