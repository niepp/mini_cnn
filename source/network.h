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

	void set_batch_size(nn_int batch_size)
	{
		for (auto &layer : m_layers)
		{
			layer->set_batch_size(batch_size);
		}
	}

	nn_float mini_batch_SGD(const varray_vec &img_vec, const varray_vec &lab_vec, const varray_vec &test_img_vec, const varray_vec &test_lab_vec
		, nn_int epoch, nn_int batch_size, nn_float learning_rate, bool calc_cost, nn_int nthreads
		, std::function<void(nn_int, nn_int)> minibatch_callback
		, std::function<void(nn_int, nn_int, nn_float, nn_float, nn_float, nn_float)> epoch_callback)
	{
		set_task_count(nthreads);

		nn_float max_accuracy = 0;
		nn_int img_count = img_vec.size();
		nn_int test_img_count = test_img_vec.size();
		nn_int batch = (img_count + batch_size - 1) / batch_size;

		nn_int img_w = img_vec[0]->width();
		nn_int img_h = img_vec[0]->height();
		nn_int img_channel = img_vec[0]->depth();
		nn_int img_size = img_w * img_h * img_channel;

		nn_int lab_w = lab_vec[0]->width();
		nn_int lab_h = lab_vec[0]->height();
		nn_int lab_channel = lab_vec[0]->depth();
		nn_int lab_size = lab_w * lab_h * lab_channel;

		std::vector<nn_int> idx_vec(img_count);
		for (nn_int k = 0; k < img_count; ++k)
		{
			idx_vec[k] = k;
		}

		for (nn_int c = 0; c < epoch; ++c)
		{
			auto tstart = get_now_ms();

			std::shuffle(idx_vec.begin(), idx_vec.end(), global_setting::m_rand_generator);

			set_batch_size(batch_size);

			for (nn_int i = 0; i < img_count; i += batch_size)
			{
				nn_int start = i;
				nn_int end = std::min<nn_int>(i + batch_size, img_count);
				varray img_batch(img_w, img_h, img_channel, batch_size);
				varray lab_batch(lab_w, lab_h, lab_channel, batch_size);
				for (nn_int j = start; j < end; ++j)
				{
					nn_int k = idx_vec[j];
					std::memcpy(&img_batch(0, 0, 0, j - start), &(*img_vec[k])[0], img_size * sizeof(nn_float));
					std::memcpy(&lab_batch(0, 0, 0, j - start), &(*lab_vec[k])[0], lab_size * sizeof(nn_float));
				}

				train_one_batch(img_batch, lab_batch);

				nn_float batch_lr = learning_rate / (end - start);
				if (!update_all_weight(batch_lr))
				{
					std::cout << "[Error] Detected infinite value in weight. stop train!" << std::endl;
				}
				minibatch_callback(end, img_count);
			}
			auto train_end = get_now_ms();
			nn_float train_elapse = (train_end - tstart) * 0.001f;
			nn_float tot_cost = calc_cost ? get_cost(img_vec, lab_vec) : (nn_float)(-1.0);
			set_batch_size(1);
			nn_int correct = test(test_img_vec, test_lab_vec);
			nn_float cur_accuracy = (1.0f * correct / test_img_count);
			max_accuracy = std::max<nn_float>(max_accuracy, cur_accuracy);
			auto test_end = get_now_ms();
			nn_float test_elapse = (test_end - train_end) * 0.001f;
			epoch_callback(c + 1, epoch, cur_accuracy, tot_cost, train_elapse, test_elapse);
		}
		return max_accuracy;
	}

	nn_int test(const varray_vec &test_img_vec, const varray_vec &test_lab_vec)
	{
		set_phase(phase_type::eTest);

		nn_assert(test_img_vec.size() == test_lab_vec.size());
		nn_int test_count = test_img_vec.size();

		nn_int correct = 0;
		for (nn_int i = 0; i < test_count; ++i)
		{
			forward(*test_img_vec[i]);
			nn_int lab = m_output_layer->get_output().arg_max();
			if (lab == test_lab_vec[i]->arg_max())
			{
				++correct;
			}
		}
		return correct;
	}

	nn_float get_cost(const varray_vec &img_vec, const varray_vec &lab_vec)
	{
		set_phase(phase_type::eTest);

		nn_int img_w = img_vec[0]->width();
		nn_int img_h = img_vec[0]->height();
		nn_int img_channel = img_vec[0]->depth();
		nn_int batch_size = img_vec[0]->count();
		nn_int img_size = img_vec[0]->img_size();

		nn_int lab_w = lab_vec[0]->width();
		nn_int lab_h = lab_vec[0]->height();
		nn_int lab_channel = lab_vec[0]->depth();
		nn_int lab_size = lab_vec[0]->img_size();

		nn_assert(img_vec.size() == lab_vec.size());
		nn_assert(img_vec[0]->count() == lab_vec[0]->count());

		nn_int img_count = img_vec.size();
		nn_float tot_cost = 0;
		for (nn_int i = 0; i < img_count; i += batch_size)
		{
			nn_int start = i;
			nn_int end = std::min<nn_int>(i + batch_size, img_count);
			nn_int bh_size = end - start;
			varray img_batch(img_w, img_h, img_channel, bh_size);
			varray lab_batch(lab_w, lab_h, lab_channel, bh_size);
			for (nn_int j = start; j < end; ++j)
			{
				std::memcpy(&img_batch(0, 0, 0, j - start), &(*img_vec[j])[0], img_size * sizeof(nn_float));
				std::memcpy(&lab_batch(0, 0, 0, j - start), &(*lab_vec[j])[0], lab_size * sizeof(nn_float));
			}
			m_input_layer->forw_prop(img_batch);
			tot_cost += bh_size * m_output_layer->calc_cost(false, lab_batch);
		}
		if (img_count > 0)
		{
			tot_cost /= img_count;
		}
		return tot_cost;
	}

	bool gradient_check(const varray &test_img, const varray &test_lab)
	{
		nn_assert(!m_layers.empty());

		set_phase(phase_type::eGradientCheck);
		set_task_count(1);
		set_batch_size(1);

		bool check_ok = true;
		for (auto &layer : m_layers)
		{
			auto &ts = layer->get_task_storage(0);
			varray &w = layer->m_w;
			varray &dw = ts.m_dw;
			nn_int w_sz = w.size();
			for (nn_int i = 0; i < w_sz; ++i)
			{
				if (!calc_gradient(test_img, test_lab, w[i], dw[i]))
				{
					check_ok = false;
				}
			}

			varray &b = layer->m_b;
			varray &db = ts.m_db;
			nn_int b_sz = b.size();
			for (nn_int i = 0; i < b_sz; ++i)
			{
				if (!calc_gradient(test_img, test_lab, b[i], db[i]))
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

	void set_phase(phase_type phase)
	{
		for (auto &layer : m_layers)
		{
			layer->set_phase_type(phase);
		}
	}

	void set_fixed_prop(nn_int task_idx)
	{
		for (auto& layer : m_layers)
		{
			layer->set_fixed_prop(task_idx);
		}
	}

	void forward(const varray& img_batch)
	{
		m_input_layer->forw_prop(img_batch);
	}

	void backward(const varray &lab_batch)
	{
		m_output_layer->back_prop(lab_batch);
	}

	bool update_all_weight(nn_float batch_lr)
	{
		for (auto &layer : m_layers)
		{
			if (!layer->update_weights(batch_lr))
			{
				return false;
			}
		}
		return true;
	}

	void train_one_batch(varray &img_batch, varray &lab_batch)
	{
		set_phase(phase_type::eTrain);
		m_input_layer->forw_prop(img_batch);
		m_output_layer->back_prop(lab_batch);
	}

	nn_int test_task(const varray_vec &test_img_vec, const varray_vec &test_lab_vec, nn_int begin, nn_int end)
	{
		set_phase(phase_type::eTest);
		nn_int c_count = 0;
		for (nn_int i = begin; i < end; ++i)
		{
			forward(*test_img_vec[i]);
			nn_int lab = m_output_layer->get_output().arg_max();
			if (lab == test_lab_vec[i]->arg_max())
			{
				++c_count;
			}
		}
		return c_count;
	}

	bool calc_gradient(const varray &test_img, const varray &test_lab, nn_float &w, nn_float &dw)
	{
		static const nn_float EPSILON = 1e-6f;
		static const nn_float Precision = 1e-4f;

		clear_all_grident();
		set_fixed_prop(0);

		nn_float prev_w = w;
		w = prev_w + EPSILON;
		m_input_layer->forw_prop(test_img);
		nn_float loss_0 = m_output_layer->calc_cost(true, test_lab);

		w = prev_w - EPSILON;
		m_input_layer->forw_prop(test_img);
		nn_float loss_1 = m_output_layer->calc_cost(true, test_lab);
		nn_float delta_by_numerical = (loss_0 - loss_1) / (nn_float(2.0) * EPSILON);

		w = prev_w;
		m_input_layer->forw_prop(test_img);
		m_output_layer->back_prop(test_lab);

		nn_float delta_by_bprop = dw;

		if (!f_is_valid(loss_0) || !f_is_valid(loss_1) || !f_is_valid(dw))
		{
			std::cout << "[overflow] loss_0:" << loss_0 << "\tloss_1:" << loss_1 << "\tdw:" << dw << std::endl;
			return false;
		}

		nn_float absError = std::abs(delta_by_bprop - delta_by_numerical);
		bool correct = absError <= Precision;
		if (!correct)
		{
		//	std::cout << "bprop:" << delta_by_bprop << "\tnumerical:" << delta_by_numerical << std::endl;
		}
		return correct;
	}

};

}

#endif //__NETWORK_H__

