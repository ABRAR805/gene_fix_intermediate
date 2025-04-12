from flask import Flask, render_template, request, jsonify, session
from utils.mutation_parser import parse_mutation
from utils.crispr_tool import generate_repair_plan
from utils.clinical_trials import get_trials
from utils.simulation import simulate_gene_editing
from utils.ai_helper import get_ai_simulation_message

app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Required to use session

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get all input values from form
        gene = request.form['gene']
        mutation = request.form['mutation']
        mut_type = request.form['mutation_type']
        disease = request.form['disease_association']
        impact = request.form['mutation_impact']
        frequency = request.form['mutation_frequency']
        gene_function = request.form['gene_function']
        family_history = request.form['family_history']
        enzyme = request.form['enzyme']
        pam = request.form['pam']
        grna = request.form['grna']
        action = request.form['action']

        parsed_mutation = parse_mutation(mutation)
        if 'error' in parsed_mutation:
            return render_template('index.html', error=parsed_mutation['error'])

        parsed_mut_type = parsed_mutation['mutation_type']
        repair_plan = generate_repair_plan(gene, mutation, enzyme, pam, grna)
        trials = get_trials(gene)

        # Store in session for animation reuse
        session.update({
            'gene': gene,
            'mutation': mutation,
            'mutation_type': mut_type,
            'disease': disease,
            'impact': impact,
            'frequency': frequency,
            'gene_function': gene_function,
            'family_history': family_history,
            'enzyme': enzyme,
            'pam': pam,
            'grna': grna
        })

        if action == 'simulate':
            simulation_results = simulate_gene_editing(gene, mutation, enzyme, pam, grna)
            return render_template('simulation_result.html',
                                   gene=gene,
                                   mutation=mutation,
                                   mut_type=parsed_mut_type,
                                   mutation_type=mut_type,
                                   disease=disease,
                                   impact=impact,
                                   frequency=frequency,
                                   gene_function=gene_function,
                                   family_history=family_history,
                                   enzyme=enzyme,
                                   pam=pam,
                                   grna=grna,
                                   repair=repair_plan,
                                   simulation=simulation_results,
                                   trials=trials,
                                   action='simulate')

        elif action == 'live_simulation':
            prompt = f'''
            take all the elements that i will be giving you to, generate a live simulaton of cancer cell reprogramming such that it should be divided into 5 phases,each phase should contain description, risk factor, the success rate (out of 1-100%),   what technology is used, what are the components are used for each step and at what amount.
            
            The following are the elements you should use:

            gene={gene}
            mutation={mutation}
            mut_type={parsed_mut_type}
            mutation_type={mut_type}
            disease={disease}
            impact={impact}
            frequency={frequency}
            gene_function={gene_function}
            family_history={family_history}
            enzyme={enzyme}
            pam={pam}
            grna={grna}

            generate a snapshot image of each phase.
            '''

            ai_message = get_ai_simulation_message(prompt)
            print(f"prompt: {prompt} \n\n ai_message: {ai_message}")

            return render_template('simulation.html',
                                   gene=gene,
                                   mutation=mutation,
                                   enzyme=enzyme,
                                   pam=pam,
                                   grna=grna,
                                   repair=repair_plan,
                                   trials=trials,
                                   action='live_simulation',
                                   ai_response=ai_message
                                   )

        else:
            return render_template('result.html',
                                   gene=gene,
                                   mutation=mutation,
                                   mut_type=parsed_mut_type,
                                   mutation_type=mut_type,
                                   disease=disease,
                                   impact=impact,
                                   frequency=frequency,
                                   gene_function=gene_function,
                                   family_history=family_history,
                                   enzyme=enzyme,
                                   pam=pam,
                                   grna=grna,
                                   repair=repair_plan,
                                   simulation=None,
                                   trials=trials,
                                   action='analyze')

    return render_template('index.html')


@app.route('/live_simulate', methods=['POST'])
def live_simulate():
    gene = session.get('gene', 'Unknown')
    mutation = session.get('mutation', 'Unknown')
    enzyme = session.get('enzyme', 'Cas9')
    pam = session.get('pam', 'NGG')
    grna = session.get('grna', 'N/A')

    # Generate simulated animation steps
    steps = [
        f"🎯 Targeting gene **{gene}** with mutation **{mutation}**",
        f"🧬 Designing gRNA: `{grna}` with PAM: `{pam}`",
        f"🔬 Enzyme `{enzyme}` binds to the target DNA site",
        f"✂️ Creating double-strand break at mutation site",
        "🧪 Inserting donor template (ssODN)",
        "🔧 Repair via HDR pathway",
        "✅ Gene mutation corrected successfully!"
    ]
    return jsonify({"steps": steps})


if __name__ == '__main__':
    app.run(debug=True)
