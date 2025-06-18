# include "modeling/elastic_iso.cuh"
# include "modeling/elastic_ani.cuh"

int main(int argc, char **argv)
{
    std::vector<Modeling *> modeling = 
    {
        new Elastic_ISO(),
        new Elastic_ANI()
    };

    auto file = std::string(argv[1]);
    auto type = std::stoi(catch_parameter("modeling_type", file));    

    modeling[type]->parameters = file;
    
    modeling[type]->set_parameters();

    auto ti = std::chrono::system_clock::now();

    for (int shot = 0; shot < modeling[type]->geometry->nrel; shot++)
    {
        modeling[type]->srcId = shot;

        modeling[type]->show_information();
        modeling[type]->time_propagation();
        modeling[type]->export_seismogram();
    }

    auto tf = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = tf - ti;
    std::cout << "\nRun time: " << elapsed_seconds.count() << " s." << std::endl;
    
    return 0;
}