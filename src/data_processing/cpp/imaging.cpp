#include "imaging.h"

namespace py = pybind11;

Imaging::Imaging(std::vector<float> x_locs, std::vector<float> y_locs, std::vector<float> z_locs, std::string radar_type, std::string aperture_type, bool is_simulation) {


	this->x_locs = x_locs;
	this->y_locs = y_locs;
	this->z_locs = z_locs;   
	auto proc_type = is_simulation ? "simulation" : "robot_collected";
    auto res = get_radar_params(radar_type, proc_type, aperture_type, "../utils/params.json");
    this->min_f = std::get<0>(res);
    this->max_f = std::get<1>(res);
    this->slope = std::get<2>(res);
    this->num_samples = std::get<3>(res);   
    this->num_rx_antenna = std::get<4>(res); 
	this->wavelength = SPEED_LIGHT / this->max_f;

}

Image Imaging::image(std::vector<Location> antenna_locs, std::vector<Measurement> measurements, std::vector<std::array<float, 3>> rx_offsets, float fft_spacing, bool apply_ti_offset, bool use_interpolated_processing) {
	std::cout << "Length antenna location: " << antenna_locs.size() << std::endl;
	std::cout << "First antenna location: " << antenna_locs[0][0] << ", " << antenna_locs[0][1] << ", " << antenna_locs[0][1] << std::endl;
	std::cout << "Number of measurements: " << measurements.size() << std::endl;
	std::cout << "Size of first measurement: " << measurements.at(0).size() << std::endl << std::endl;
	Image p_xyz(x_locs.size(), std::vector<std::vector<std::complex<float>>>(y_locs.size(), std::vector<std::complex<float>>(z_locs.size())));

	const uint totalIterations = x_locs.size() * y_locs.size() * z_locs.size();
    uint completedIterations = 0;

	#pragma omp parallel for
	for (uint x = 0; x < x_locs.size(); x++) {

		float x_loc = x_locs[x];
		for (uint y = 0; y < y_locs.size(); y++) {

			float y_loc = y_locs[y];
			for (uint z = 0; z < z_locs.size(); z++) {

				float z_loc = z_locs[z];
				std::complex<float> sum = std::complex<float>(0,0);
				uint used_antennas = 0;
				for (uint i = 0; i < antenna_locs.size(); i++) {
					used_antennas++;

					Location antenna_loc = antenna_locs.at(i);
					float antenna_x_diff = x_loc - antenna_loc[0];
					float antenna_y_diff = y_loc - antenna_loc[1];
					float antenna_z_diff = z_loc - antenna_loc[2];
					float forward_dist = std::sqrt(antenna_x_diff * antenna_x_diff + 
												   antenna_y_diff * antenna_y_diff + 
												   antenna_z_diff * antenna_z_diff);
					if (apply_ti_offset){
						forward_dist += 0.15;
					}


					for (uint k = 0; k < this->num_rx_antenna; k++) {
						float rx_offset_x = rx_offsets[k][0];
						float rx_offset_y = rx_offsets[k][1];
						float rx_offset_z = rx_offsets[k][2];
						float rx_x_diff = x_loc - (antenna_loc[0] + rx_offset_x);
						float rx_y_diff = y_loc - (antenna_loc[1] + rx_offset_y);
						float rx_z_diff = z_loc - (antenna_loc[2] + rx_offset_z);
						float back_dist = std::sqrt(rx_x_diff * rx_x_diff + rx_y_diff * rx_y_diff + rx_z_diff * rx_z_diff);

						float distance = forward_dist + back_dist;



						if (use_interpolated_processing ){ // Apply faster but approximate coorelation
							// Check that our distance is valid
							if (distance < 0 || distance > fft_spacing*this->num_samples) {
								continue;
							}

							// Find which bin within the range FFT this distance falls
							int dist_bin = floorf(distance / fft_spacing/2);

							// Select the appropriate measurement, and coorelate with the AoA phase
							sum += measurements[i][dist_bin][k] * std::exp(std::complex<float>(0., -2. * M_PI * distance / this->wavelength));
						} else {
							for (uint j = 0; j < this->num_samples; j++) {
								sum += measurements[i][j][k] * std::exp(std::complex<float>(0., -2. * M_PI * distance * j * this->slope / SPEED_LIGHT)) * 
									std::exp(std::complex<float>(0., -2. * M_PI * distance / this->wavelength));
							}
						}
					}

				}
				p_xyz[x][y][z] = sum / std::complex<float>(used_antennas,0);

				// Progress bar
				#pragma omp critical
				{
					completedIterations++;
					float progress = static_cast<float>(completedIterations) / totalIterations * 100;
					std::cout << "\rProgress: " << std::fixed << std::setprecision(2) << progress << "%";
					std::cout.flush();
				}

			}

		}

	}  

    std::cout << "\rProgress: 100.00%" << std::endl;
	return p_xyz;

}


PYBIND11_MODULE(imaging, m) {  
	py::class_<Imaging>(m,"Imaging")
		.def(py::init<std::vector<float>, std::vector<float>, std::vector<float>, std::string, std::string, bool>())
		.def("image", &Imaging::image);
}
