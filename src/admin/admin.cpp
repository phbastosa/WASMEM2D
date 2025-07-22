# include "admin.hpp"

bool str2bool(std::string s)
{
    bool b;

    std::for_each(s.begin(), s.end(), [](char & c){c = ::tolower(c);});
    std::istringstream(s) >> std::boolalpha >> b;

    return b;
}

void import_binary_float(std::string path, float * array, int n)
{
    std::ifstream file(path, std::ios::in);

    if (!file.is_open())
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");
    
    file.read((char *) array, n * sizeof(float));
    
    file.close();    
}

void export_binary_float(std::string path, float * array, int n)
{
    std::ofstream file(path, std::ios::out);
    
    if (!file.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");

    file.write((char *) array, n * sizeof(float));

    std::cout<<"\nBinary file \033[34m" + path + "\033[0;0m was successfully written."<<std::endl;

    file.close();
}

void import_text_file(std::string path, std::vector<std::string> &elements)
{
    std::ifstream file(path, std::ios::in);
    
    if (!file.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + path + "\033[0;0m could not be opened!");

    std::string line;
    while(getline(file, line))
        if (line[0] != '#') elements.push_back(line);

    file.close();
}

std::string catch_parameter(std::string target, std::string file)
{
    char spaces = ' ';
    char comment = '#';

    std::string line;
    std::string variable;

    std::ifstream parameters(file);

    if (!parameters.is_open()) 
        throw std::invalid_argument("Error: \033[31m" + file + "\033[0;0m could not be opened!");

    while (getline(parameters, line))
    {           
        if ((line.front() != comment) && (line.front() != spaces))        
        {
            if (line.find(target) == 0)
            {
                for (int i = line.find("=")+2; i < line.size(); i++)
                {    
                    if (line[i] == '#') break;
                    variable += line[i];            
                }

                break;
            }
        }                 
    }
    
    parameters.close();

    variable.erase(remove(variable.begin(), variable.end(), ' '), variable.end());

    return variable;
}

std::vector<std::string> split(std::string s, char delimiter)
{
    std::string token;
    std::vector<std::string> tokens;
    std::istringstream tokenStream(s);

    while (getline(tokenStream, token, delimiter)) 
        tokens.push_back(token);
   
    return tokens;
}

float sinc(float x) 
{
    if (fabsf(x) < 1e-8f) return 1.0f;
    return sinf(M_PI * x) / (M_PI * x);
}

float bessel_i0(float x) 
{
    float sum = 1.0f, term = 1.0f, k = 1.0f;
    
    while (term > 1e-10f) 
    {
        term *= (x / (2.0f * k)) * (x / (2.0f * k));
        sum += term;
        k += 1.0f;
    }

    return sum;
}

std::vector<std::vector<float>> kaiser_weights(float x, float z, int ix0, int iz0, float dx, float dz, float beta) 
{
    const int N = 4;
    float sum = 0.0f;

    std::vector<std::vector<float>> weights(N, std::vector<float>(N));

    float rmax = 1.5f * sqrtf(2.0f) * std::max(dx, dz);
    float I0_beta = bessel_i0(beta);

    for (int i = 0; i < N; ++i) 
    {    
        float zi = (iz0 + i - 1) * dz;
        float dzr = (z - zi) / dz;
        
        for (int j = 0; j < N; ++j) 
        {
            float xj = (ix0 + j - 1) * dx;
            float dxr = (x - xj) / dx;

            float rz = z - zi;
            float rx = x - xj;
            float r = sqrtf(rx * rx + rz * rz);
            float rnorm = 2.0 * r / rmax;

            float wij = 0.0f;
            if (rnorm <= 1.0f) 
            {
                float arg = beta * sqrtf(1.0f - rnorm * rnorm);
                wij = bessel_i0(arg) / I0_beta;
            }

            float sinc_term = sinc(dxr) * sinc(dzr);

            weights[i][j] = sinc_term * wij;

            sum += wij;
        }
    }

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            weights[i][j] /= sum;

    return weights;
}
