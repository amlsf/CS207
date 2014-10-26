#include "CS207/Util.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
/** Return true iff @a n is prime.
 * @pre @a n >= 0
 */

bool is_prime(int n){
  //Vector to store all the known prime numbers
  static std::vector<int> primes;
  assert(n >= 0);
  
  //Check to see if current n is already a known prime number
  if(std::binary_search(primes.begin(), primes.end(), n)){
    return true;
  }

  //Check to see if n is divisble by prior known prime numbers
  for(unsigned k = 0; k < primes.size(); k++){
    if(n % primes[k] == 0){
      return false;
    }
  }

  //If n is not in primes, need to run brute force
  for(int i = 2; i <= (int)sqrt(n); i++){
    if(n % i == 0){
      return false;
    }
  }
  //New prime, add to primes
  primes.push_back(n);
  //Sort primes as it is a pre-condition needed for binary search
  sort(primes.begin(), primes.end());
  return true;
}


int main(){
  while (!std::cin.eof()) {
    // How many primes to test? And should we print them?
    std::cerr << "Input Number: ";
    int n = 0;
    CS207::getline_parsed(std::cin, n);
    if (n <= 0)
      break;

    std::cerr << "Print Primes (y/n): ";
    char confirm = 'n';
    CS207::getline_parsed(std::cin, confirm);
    bool print_primes = (confirm == 'y' || confirm == 'Y');
    CS207::Clock timer;

    // Loop and count primes from 2 up to n
    int num_primes = 0;
    for (int i = 2; i <= n; ++i) {
      if (is_prime(i)) {
        ++num_primes;
        if (print_primes)
          std::cout << i << std::endl;
      }
    }

    double elapsed_time = timer.seconds();
    std::cout << "There are " << num_primes
              << " primes less than or equal to " << n << ".\n"
              << "Found in " << (1000 * elapsed_time) << " milliseconds.\n\n";
  }
  return 0;
}