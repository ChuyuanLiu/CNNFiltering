namespace tfAOT
{
    class model
    {
        public:
            model(int threads = 1);
            void run(float *input_hit, float *input_info, float *output);
        private:
            void *GraphPtr;
    };
}